#!/usr/bin/env python3
"""
Анализ энергий и вторичной структуры (DSSP) из Excel таблиц и PDB файлов.
Расширенная версия: учитываются все 8 типов вторичной структуры (H, E, B, T, S, G, I, C).
"""

import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
from datetime import datetime

# ====================== ФУНКЦИИ ======================

def load_energies_from_excel(excel_path, sheet_name=0):
    """Загружает энергии из Excel файла."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.iloc[:, [1, 2]].dropna()
    df.columns = ['Filename', 'Energy']
    df['Filename'] = df['Filename'].astype(str).str.strip()
    df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')
    df = df.dropna()
    return df

def parse_filename(filename):
    """Извлекает систему и номер модели из названия файла."""
    system_match = re.search(r'([\dA-Z]+WT|[\dA-Z]+MUT)', filename)
    rank_match = re.search(r'_rank_(\d+)', filename)
    if system_match and rank_match:
        system = system_match.group(1)
        rank_num = int(rank_match.group(1))
        return system, rank_num
    return None, None

def find_pdb_file(system, rank_number, base_dir):
    """Находит PDB файл по системе и номеру модели."""
    patterns = [
        os.path.join(base_dir, f"relaxed_{system}_unrelaxed_rank_{rank_number:03d}*.pdb"),
        os.path.join(base_dir, f"relaxed_{system}_rank_{rank_number:03d}*.pdb"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    sub_dir = os.path.join(base_dir, system)
    if os.path.isdir(sub_dir):
        patterns = [
            os.path.join(sub_dir, f"relaxed_{system}_unrelaxed_rank_{rank_number:03d}*.pdb"),
            os.path.join(sub_dir, f"relaxed_{system}_rank_{rank_number:03d}*.pdb"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
    return None

# НОВОЕ: полный список типов вторичной структуры
SS_TYPES = ['H', 'E', 'B', 'T', 'S', 'G', 'I', 'C']   # C = coil (пробел в DSSP)
SS_DISPLAY = {'H':'α-спираль', 'E':'β-лист', 'B':'β-мост', 'T':'Поворот',
              'S':'Изгиб', 'G':'3₁₀-спираль', 'I':'π-спираль', 'C':'Неупорядоченный виток'}

def compute_dssp_fractions(pdb_path):
    """
    Загружает PDB, запускает DSSP, возвращает словарь с долями всех типов.
    Возвращает dict {type: percent} или None при ошибке.
    """
    try:
        traj = md.load(pdb_path)
        ss_codes = md.compute_dssp(traj, simplified=False)[0]   # массив символов
        total = len(ss_codes)
        if total == 0:
            return {ss_type: 0.0 for ss_type in SS_TYPES}
        counts = {ss_type: 0 for ss_type in SS_TYPES}
        for code in ss_codes:
            # DSSP возвращает пробел для coil – заменяем на 'C'
            if code == ' ':
                code = 'C'
            if code in counts:
                counts[code] += 1
            else:
                # на случай нестандартного символа – отнесём к coil
                counts['C'] += 1
        percentages = {ss_type: (counts[ss_type] / total * 100.0) for ss_type in SS_TYPES}
        return percentages
    except Exception as e:
        print(f"  ⚠ Ошибка при обработке {pdb_path}: {e}")
        return None

def process_system_directory(system_dir, results_list):
    """Обрабатывает директорию с системой."""
    system_name = os.path.basename(system_dir.rstrip('/'))
    print(f"\n📁 Обработка системы: {system_name}")

    excel_files = glob.glob(os.path.join(system_dir, "*.xlsx"))
    if not excel_files:
        print(f"   ⚠ Excel файл не найден в {system_name}")
        return 0
    excel_file = excel_files[0]
    print(f"   📊 Файл: {os.path.basename(excel_file)}")

    energy_df = load_energies_from_excel(excel_file)
    print(f"   ✓ Загружено {len(energy_df)} записей")

    energy_df[['System', 'RankNumber']] = energy_df['Filename'].apply(
        lambda x: pd.Series(parse_filename(x))
    )
    energy_df = energy_df.dropna(subset=['System', 'RankNumber'])
    energy_df['RankNumber'] = energy_df['RankNumber'].astype(int)
    print(f"   ✓ Распарсено {len(energy_df)} записей")

    energy_df['PDB_Path'] = energy_df.apply(
        lambda row: find_pdb_file(row['System'], row['RankNumber'], system_dir),
        axis=1
    )
    not_found = energy_df[energy_df['PDB_Path'].isna()]
    if len(not_found) > 0:
        print(f"   ⚠ Не найдены PDB для {len(not_found)} записей")
    energy_df = energy_df.dropna(subset=['PDB_Path'])

    if len(energy_df) == 0:
        print(f"   ❌ Нет PDB файлов для обработки")
        return 0

    print(f"   🔬 Расчёт DSSP для {len(energy_df)} моделей...")
    processed_count = 0
    for idx, row in tqdm(energy_df.iterrows(), total=len(energy_df), leave=False):
        fractions = compute_dssp_fractions(row['PDB_Path'])
        if fractions is not None:
            # НОВОЕ: добавляем все проценты в результат
            entry = {
                'System': row['System'],
                'RankNumber': row['RankNumber'],
                'Filename': row['Filename'],
                'Energy': row['Energy'],
                'PDB_Path': row['PDB_Path']
            }
            entry.update(fractions)   # добавить ключи H, E, B, T, S, G, I, C
            results_list.append(entry)
            processed_count += 1

    print(f"   ✓ Обработано {processed_count} моделей")
    return processed_count

# ======================== ГЛАВНЫЙ БЛОК ========================
if __name__ == '__main__':
    # 1. Получаем рабочую директорию
    if len(sys.argv) < 2:
        print("Использование: python DSSP.py /path/to/working/directory")
        print("\nВведите путь к рабочей директории:")
        work_dir = input().strip()
    else:
        work_dir = sys.argv[1]

    if not os.path.isdir(work_dir):
        print(f"❌ Ошибка: директория '{work_dir}' не найдена")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"АНАЛИЗ ЭНЕРГИЙ И ВСЕХ ТИПОВ ВТОРИЧНОЙ СТРУКТУРЫ (DSSP)")
    print(f"{'='*70}")
    print(f"Рабочая директория: {work_dir}\n")

    output_dir = os.path.join(work_dir, "DSSP_Results")
    plots_by_system_dir = os.path.join(output_dir, "plots_by_system")
    os.makedirs(plots_by_system_dir, exist_ok=True)
    print(f"✓ Создана директория: {output_dir}\n")

    # Поиск систем
    system_dirs = []
    for item in sorted(os.listdir(work_dir)):
        item_path = os.path.join(work_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item != 'DSSP_Results':
            if glob.glob(os.path.join(item_path, "*.xlsx")):
                system_dirs.append(item_path)

    if not system_dirs:
        print("❌ Ошибка: не найдены папки с .xlsx файлами")
        sys.exit(1)

    print(f"Найдено {len(system_dirs)} систем:\n")
    for d in system_dirs:
        print(f"  • {os.path.basename(d)}")
    print()

    # Обработка
    all_results = []
    total_processed = 0
    for system_dir in system_dirs:
        processed = process_system_directory(system_dir, all_results)
        total_processed += processed

    df_final = pd.DataFrame(all_results)

    if len(df_final) == 0:
        print("\n❌ Ошибка: не удалось обработать ни один PDB файл")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"✓ Всего обработано {len(df_final)} моделей\n")

    # Сохраняем CSV со всеми типами
    output_csv = os.path.join(output_dir, "energy_dssp_analysis.csv")
    df_final.to_csv(output_csv, index=False)
    print(f"✓ Сохранена таблица: energy_dssp_analysis.csv\n")

    # Статистика и корреляции (для каждого типа)
    summary_text = []
    summary_text.append("="*70)
    summary_text.append("СТАТИСТИКА И КОРРЕЛЯЦИИ (все типы вторичной структуры)")
    summary_text.append("="*70)
    summary_text.append(f"\nВремя анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_text.append(f"Всего моделей обработано: {len(df_final)}\n")

    print("="*70)
    print("СТАТИСТИКА ПО СИСТЕМАМ (среднее ± std)")
    print("="*70)

    for system in sorted(df_final['System'].unique()):
        sub = df_final[df_final['System'] == system]
        stats_text = f"\n{system} (N={len(sub)}):"
        print(stats_text)
        summary_text.append(stats_text)

        # Энергия
        mean_e = sub['Energy'].mean()
        std_e = sub['Energy'].std()
        stat_line = f"  {'Energy':<20} {mean_e:6.1f} ± {std_e:6.1f} kcal/mol"
        print(stat_line)
        summary_text.append(stat_line)

        # Каждый тип вторичной структуры
        for ss_type in SS_TYPES:
            mean_val = sub[ss_type].mean()
            std_val = sub[ss_type].std()
            stat_line = f"  {SS_DISPLAY[ss_type]:<20} {mean_val:6.1f} ± {std_val:6.1f} %"
            print(stat_line)
            summary_text.append(stat_line)

    print("\n" + "="*70)
    print("КОРРЕЛЯЦИИ С ЭНЕРГИЕЙ (коэф. Спирмена, все системы)")
    print("="*70)
    summary_text.append("\n" + "="*70)
    summary_text.append("КОРРЕЛЯЦИИ С ЭНЕРГИЕЙ (Спирмен)")
    summary_text.append("="*70)

    for ss_type in SS_TYPES:
        corr, p_val = spearmanr(df_final[ss_type], df_final['Energy'])
        sig = "✓" if p_val < 0.05 else "✗"
        corr_line = f"{SS_DISPLAY[ss_type]:<15}: r = {corr:.3f}, p = {p_val:.4e} {sig}"
        print(corr_line)
        summary_text.append(corr_line)

    # Подготовка к графикам
    systems = sorted(df_final['System'].unique())
    palette = sns.color_palette("husl", len(systems))
    color_dict = dict(zip(systems, palette))

    print("\n" + "="*70)
    print("СОЗДАНИЕ ГРАФИКОВ (subplots для всех типов)")
    print("="*70)

    # ---- 1. Общий график: Energy vs каждый тип (subplots 2×4) для всех систем ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    for idx, ss_type in enumerate(SS_TYPES):
        ax = axes[idx]
        for system in systems:
            sub = df_final[df_final['System'] == system]
            marker = 'o' if 'WT' in system else 's'
            ax.scatter(sub[ss_type], sub['Energy'],
                       label=system, color=color_dict[system], marker=marker,
                       s=60, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(f'{SS_DISPLAY[ss_type]} (%)', fontsize=11)
        ax.set_ylabel('Energy (kcal/mol)', fontsize=11)
        ax.set_title(f'{SS_DISPLAY[ss_type]}', fontsize=12)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(alpha=0.2, linestyle='--')
    # Общая легенда
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    fig.suptitle('Зависимость энергии связывания от типов вторичной структуры\n(все системы)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])
    plot_all = os.path.join(output_dir, 'plot_all_systems_Energy_vs_SS_types.jpg')
    plt.savefig(plot_all, dpi=600, format='jpg', bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранён общий график: plot_all_systems_Energy_vs_SS_types.jpg")

    # ---- 2. Графики по каждой системе (Energy vs каждый тип) ----
    print("📈 Создание графиков для каждой системы...")
    for system in systems:
        sub = df_final[df_final['System'] == system]
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        for idx, ss_type in enumerate(SS_TYPES):
            ax = axes[idx]
            ax.scatter(sub[ss_type], sub['Energy'],
                       color=color_dict[system], s=80, alpha=0.7,
                       edgecolor='black', linewidth=0.7)
            ax.set_xlabel(f'{SS_DISPLAY[ss_type]} (%)', fontsize=11)
            ax.set_ylabel('Energy (kcal/mol)', fontsize=11)
            ax.set_title(f'{SS_DISPLAY[ss_type]}', fontsize=12)
            ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(alpha=0.2, linestyle='--')
        fig.suptitle(f'{system}: Энергия vs вторичная структура (N={len(sub)})',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(plots_by_system_dir, f'{system}_Energy_vs_SS_types.jpg')
        plt.savefig(plot_path, dpi=600, format='jpg', bbox_inches='tight')
        plt.close()
        print(f"   ✓ {system}")

    # ---- 3. (Опционально) Pairplot всех числовых переменных ----
    try:
        # Выбираем числовые колонки: Energy + все типы
        numeric_cols = ['Energy'] + SS_TYPES
        pairplot_data = df_final[numeric_cols + ['System']].copy()
        # Переименуем для наглядности
        rename_dict = {'Energy': 'Energy (kcal/mol)'}
        rename_dict.update({ss: SS_DISPLAY[ss] for ss in SS_TYPES})
        pairplot_data = pairplot_data.rename(columns=rename_dict)
        g = sns.pairplot(pairplot_data, hue='System', palette=color_dict,
                         diag_kind='hist', markers=['o','s'], plot_kws={'alpha':0.6})
        g.fig.suptitle('Попарные корреляции энергии и типов вторичной структуры', y=1.02)
        pairplot_path = os.path.join(output_dir, 'pairplot_all_variables.jpg')
        g.savefig(pairplot_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ Сохранён pairplot: pairplot_all_variables.jpg")
    except Exception as e:
        print(f"⚠ Не удалось построить pairplot: {e}")

    # Сохраняем summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_text))

    print(f"\n✓ Сохранён summary: summary.txt")

    print("\n" + "="*70)
    print("АНАЛИЗ ЗАВЕРШЁН ✓")
    print("="*70)
    print(f"\nРезультаты сохранены в: {output_dir}")
    print(f"  ├── energy_dssp_analysis.csv")
    print(f"  ├── plot_all_systems_Energy_vs_SS_types.jpg")
    print(f"  ├── pairplot_all_variables.jpg")
    print(f"  ├── plots_by_system/")
    for system in systems[:3]:
        print(f"  │   ├── {system}_Energy_vs_SS_types.jpg")
    if len(systems) > 3:
        print(f"  │   └── ... (ещё {len(systems)-3} систем)")
    print(f"  └── summary.txt")
    print(f"\n📊 Обработано систем: {len(systems)}")
    print(f"📈 Обработано моделей: {len(df_final)}")
    print()