#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Pipeline v6: PDB → Membrane Alignment → Implicit Solvent+Membrane MD → Entropy + ΔG
Добавлено: автоматическое сканирование n_states (--scan_ns) и расчёт G_conf = H - T*S
"""
from __future__ import annotations
import os, gc, math, glob, argparse, time, traceback, warnings
import numpy as np
import pandas as pd
import mdtraj as md
import lzma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
import openmm as mm
from openmm import app, unit
from openmm.unit import kelvin, picosecond, nanometer, femtosecond
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# ================= НАСТРОЙКИ ОКРУЖЕНИЯ =================
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OMP_NUM_THREADS'] = '1'

CFG = {
    "temp_K": 300.0, "friction_ps": 1.0,
    # Основной шаг интегрирования (production и основная equil)
    "dt_fs": 2.0,
    # “Мягкий старт” (первые шаги equil на меньшем dt)
    "dt_fs_warmup": 1.0,
    "equil_warmup_steps": 5000,
    "sim_ps": 500.0, "save_ps": 0.5,
    "n_states": 128, "lzma_preset": 4,
    "restraint_k": 50.0, "platform": "CPU",
    "min_steps": 5000, "equil_steps": 50000,
    "console_report_steps": 5000,
    "nan_check_steps": 1000,
    "window_frames": 100, "step_frames": 20,
    "output_dir": "",
    "hydrophobic_residues": {"ALA","VAL","LEU","ILE","PHE","TRP","MET","CYS"},
    "membrane_k": 100.0, "membrane_half_width": 1.5,
    "scan_ns": False,  # 🔍 Новый флаг
    "ns_range": [64, 96, 128, 160, 192, 256]  # 🔍 Диапазон для сканирования
}

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
def clean_and_align_pdb(pdb_path: str, out_pdb: str) -> str:
    traj = md.load(pdb_path)
    protein_atoms = [a.index for a in traj.topology.atoms if a.residue.is_protein]
    if not protein_atoms: raise ValueError("В PDB не найдено белковых атомов.")
    traj = traj.atom_slice(protein_atoms)
    
    hydro_idx = [a.index for a in traj.topology.atoms 
                 if a.name == 'CA' and a.residue.name in CFG["hydrophobic_residues"]]
                 
    if len(hydro_idx) >= 5:
        coords_ref = traj.xyz[0, hydro_idx]
        com = coords_ref.mean(axis=0)
        U, S, Vt = np.linalg.svd(coords_ref - com)
        axis = Vt[0] / np.linalg.norm(Vt[0])
        z_axis = np.array([0, 0, 1])
        rot_vec = np.cross(axis, z_axis)
        rot = R.from_rotvec(rot_vec) if np.linalg.norm(rot_vec) > 1e-6 else R.identity()
        
        n_frames, n_atoms, _ = traj.xyz.shape
        flat_centered = (traj.xyz - com).reshape(-1, 3)
        traj.xyz = rot.apply(flat_centered).reshape(n_frames, n_atoms, 3) + com
    else:
        print(f"  ⚠️ Мало гидрофобных остатков ({len(hydro_idx)}), центрирую без вращения.")
        traj.xyz -= traj.xyz.mean(axis=1, keepdims=True)
    traj[0].save(out_pdb)
    return out_pdb

def detect_peptide_chain(pdb_path: str) -> str:
    top = md.load_topology(pdb_path)
    chains = {}
    for chain in top.chains:
        n_res = sum(1 for r in chain.residues if r.is_protein)
        if n_res > 0: chains[chain.chain_id] = n_res
    if not chains: raise ValueError("Нет белковых цепей.")
    return min(chains, key=chains.get)

def compress_size(data: np.ndarray, preset: int = 4) -> int:
    return len(lzma.compress(data.tobytes(), preset=preset))

def calc_entropy_window(dihedrals: np.ndarray, ns: int = 128, preset: int = 4) -> tuple[float, dict]:
    n_frames, n_dof = dihedrals.shape
    if n_frames == 0: return 0.0, {}
    rng = dihedrals.max(axis=0) - dihedrals.min(axis=0)
    rng[rng == 0] = 1.0
    disc = np.clip(np.floor(((dihedrals - dihedrals.min(axis=0)) / rng) * ns), 0, ns-1).astype(np.uint8)
    flat = disc.ravel()
    Cd = compress_size(flat, preset)
    C0 = compress_size(np.zeros_like(flat), preset)
    C1 = compress_size(np.random.randint(0, ns, len(flat), dtype=np.uint8), preset)
    debug = {"Cd": Cd, "C0": C0, "C1": C1}
    if C1 <= C0:
        debug["error"] = "Bad calibration (C1 <= C0)"
        return 0.0, debug
    eta = np.clip((Cd - C0) / (C1 - C0), 0.0, 1.0)
    debug["eta"] = eta
    return eta * n_dof * math.log(ns), debug

def find_optimal_ns(dihedrals: np.ndarray, ns_range: list, preset: int) -> tuple[int, float]:
    """Сканирует ns_range и возвращает ns с минимальной SA (Fig. 3d статьи)."""
    best_ns, best_SA = ns_range[0], float('inf')
    for ns in ns_range:
        SA, dbg = calc_entropy_window(dihedrals, ns, preset)
        if np.isfinite(SA) and SA < best_SA:
            best_SA, best_ns = SA, ns
    return best_ns, best_SA

# ================= МОЛЕКУЛЯРНАЯ ДИНАМИКА =================
def run_implicit_md(pdb_path: str, out_dir: str, cfg: dict, pep_chain: str) -> tuple[str, str]:
    pdb = app.PDBFile(pdb_path)
    
    # В разных версиях OpenMM/наборов ff аргумент implicitSolvent может отсутствовать
    # и/или implicit solvent подключается отдельным XML (amber14/implicit/*).
    system = None
    last_err: Exception | None = None

    print("  ▶️ MD: building implicit-solvent system...", flush=True)

    # Путь №1: старый API (implicitSolvent=...) — попробуем сначала.
    try:
        ff = app.ForceField("amber99sb.xml")
        system = ff.createSystem(
            pdb.topology,
            implicitSolvent=app.OBC2,
            nonbondedMethod=app.NoCutoff,  # GB-модели требуют NoCutoff
            soluteDielectric=1.0,
            solventDielectric=78.5,
            constraints=app.HBonds,
        )
    except Exception as e:
        last_err = e

    # Путь №2: новый/рекомендуемый способ — явный implicit XML, без implicitSolvent kwarg.
    if system is None:
        candidates = [
            # Комбинированные XML (в некоторых сборках это самый простой вариант)
            ("amber99_obc.xml",),
            ("amber03_obc.xml",),
            ("amber10_obc.xml",),
            # Раздельно: белок + implicit GBSA (структура каталогов в OpenMM: implicit/*.xml)
            ("amber14/protein.ff14SB.xml", "implicit/obc2.xml"),
            ("amber14/protein.ff14SB.xml", "implicit/gbn2.xml"),
            ("amber14-all.xml", "implicit/obc2.xml"),
            ("amber14-all.xml", "implicit/gbn2.xml"),
            ("amber99sb.xml", "implicit/obc2.xml"),
            ("amber99sb.xml", "implicit/gbn2.xml"),
        ]
        for cand in candidates:
            try:
                ff = app.ForceField(*cand)
                system = ff.createSystem(
                    pdb.topology,
                    nonbondedMethod=app.NoCutoff,
                    soluteDielectric=1.0,
                    solventDielectric=78.5,
                    constraints=app.HBonds,
                )
                break
            except Exception as e:
                last_err = e

    if system is None:
        raise RuntimeError(
            "Не удалось создать систему для implicit solvent.\n"
            "Чаще всего причина — в установке OpenMM нет нужных forcefield XML для GBSA\n"
            "или используется API, где implicit solvent задаётся только через отдельный XML.\n"
            f"Последняя ошибка: {type(last_err).__name__}: {last_err}"
        ) from last_err

    print("  ▶️ MD: system built.", flush=True)
        
    all_atoms = list(pdb.topology.atoms())
    ca_ids = [a.index for a in all_atoms if a.name == 'CA' and a.residue.chain.id != pep_chain]
    if ca_ids:
        frc = mm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        frc.addGlobalParameter("k", cfg["restraint_k"])
        frc.addPerParticleParameter("x0"); frc.addPerParticleParameter("y0"); frc.addPerParticleParameter("z0")
        for idx in ca_ids:
            pos = pdb.positions[idx]
            frc.addParticle(idx, [pos[0]._value, pos[1]._value, pos[2]._value])
        system.addForce(frc)
        
    hydro_ids = [a.index for a in all_atoms if a.residue.name in cfg["hydrophobic_residues"]]
    if hydro_ids:
        mem_force = mm.CustomExternalForce("k_mem * max(0, abs(z) - w_half)^2")
        mem_force.addGlobalParameter("k_mem", cfg["membrane_k"])
        mem_force.addGlobalParameter("w_half", cfg["membrane_half_width"])
        for idx in hydro_ids: mem_force.addParticle(idx, [])
        system.addForce(mem_force)

    # Начинаем equil с более маленьким dt, потом переключаемся на основной.
    dt_warmup_fs = float(cfg.get("dt_fs_warmup", cfg["dt_fs"]))
    dt_main_fs = float(cfg["dt_fs"])
    integrator = mm.LangevinMiddleIntegrator(
        cfg["temp_K"] * kelvin,
        cfg["friction_ps"] / picosecond,
        dt_warmup_fs * femtosecond,
    )

    # Платформа: CUDA в WSL/без GPU часто падает на инициализации (CUDA error 100),
    # поэтому делаем прозрачный fallback на CPU.
    platform_name = cfg.get("platform", "CPU")
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        sim = app.Simulation(pdb.topology, system, integrator, platform)
    except Exception as e:
        if platform_name == "CUDA":
            print(f"  ⚠️ CUDA не стартовала ({type(e).__name__}). Переключаюсь на CPU.")
            platform = mm.Platform.getPlatformByName("CPU")
            sim = app.Simulation(pdb.topology, system, integrator, platform)
        else:
            raise
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(cfg["temp_K"]*kelvin)

    log_path = os.path.join(out_dir, "md.log")
    dcd_path = os.path.join(out_dir, "traj.dcd")

    # Репортёры добавляем ДО equilibration, чтобы файлы появлялись сразу
    interval = int(cfg["save_ps"] / (cfg["dt_fs"] / 1000))
    sim.reporters.append(app.DCDReporter(dcd_path, interval))
    sim.reporters.append(
        app.StateDataReporter(
            log_path,
            interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
        )
    )
    # “Живой” прогресс в консоль (отдельно от файла лога)
    console_steps = int(cfg.get("console_report_steps", 0) or 0)
    if console_steps > 0:
        sim.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                console_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                temperature=True,
                speed=True,
                totalSteps=int(cfg["equil_steps"] + (cfg["sim_ps"] / (cfg["dt_fs"] / 1000))),
            )
        )

    print("  ▶️ MD: minimization...", flush=True)
    sim.minimizeEnergy(maxIterations=cfg["min_steps"])

    def _positions_have_nan() -> bool:
        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        return bool(np.isnan(pos.value_in_unit(nanometer)).any())

    nan_check = int(cfg.get("nan_check_steps", 0) or 0)
    if nan_check <= 0:
        nan_check = int(cfg.get("console_report_steps", 1000) or 1000)

    print("  ▶️ MD: equilibration...", flush=True)
    equil_total = int(cfg["equil_steps"])
    warmup_steps = int(cfg.get("equil_warmup_steps", 0) or 0)
    warmup_steps = max(0, min(warmup_steps, equil_total))

    if warmup_steps > 0:
        print(f"  ▶️ MD: warmup equil ({warmup_steps} steps @ {dt_warmup_fs:g} fs)...", flush=True)
        remaining = warmup_steps
        while remaining > 0:
            n = min(nan_check, remaining)
            sim.step(n)
            remaining -= n
            if _positions_have_nan():
                raise ValueError(
                    "Particle position is NaN during warmup equilibration. "
                    "Попробуйте уменьшить dt_fs_warmup, увеличить friction, "
                    "и/или увеличить min_steps."
                )

    if dt_main_fs != dt_warmup_fs:
        integrator.setStepSize(dt_main_fs * femtosecond)
        print(f"  ▶️ MD: switching dt to {dt_main_fs:g} fs.", flush=True)

    remaining = equil_total - warmup_steps
    while remaining > 0:
        n = min(nan_check, remaining)
        sim.step(n)
        remaining -= n
        if _positions_have_nan():
            raise ValueError(
                "Particle position is NaN during equilibration. "
                "Попробуйте уменьшить dt, увеличить friction, "
                "и/или увеличить min_steps."
            )

    print("  ▶️ MD: production...", flush=True)
    prod_steps = int(cfg["sim_ps"] / (cfg["dt_fs"] / 1000))
    remaining = prod_steps
    while remaining > 0:
        n = min(nan_check, remaining)
        sim.step(n)
        remaining -= n
        if _positions_have_nan():
            raise ValueError(
                "Particle position is NaN during production. "
                "Попробуйте уменьшить dt (CFG['dt_fs']=1.0), увеличить friction, "
                "и/или увеличить min_steps."
            )
    
    del sim.context, sim.integrator, sim; gc.collect()
    return dcd_path, log_path

# ================= АНАЛИЗ ЭНТРОПИИ =================
def analyze_single_complex(complex_id: str, pdb_path: str, cfg: dict) -> dict:
    cdir = os.path.join(cfg["output_dir"], complex_id)
    dcd = os.path.join(cdir, "traj.dcd")
    log = os.path.join(cdir, "md.log")
    if not os.path.exists(dcd) or not os.path.exists(log):
        return {"complex_id": complex_id, "error": "Missing DCD/LOG"}
        
    try:
        traj = md.load(dcd, top=pdb_path)
        pep_chain = detect_peptide_chain(pdb_path)
        pep_indices = [a.index for a in traj.topology.atoms if a.residue.chain.id == pep_chain]
        if not pep_indices: return {"complex_id": complex_id, "error": "No peptide atoms"}
            
        traj_pep = traj.atom_slice(pep_indices)
        phis = md.compute_phi(traj_pep)[0]
        psis = md.compute_psi(traj_pep)[0]
        dihedrals = np.hstack([phis, psis])
        
        df_log = pd.read_csv(log, skipinitialspace=True)
        df_log.columns = [c.strip() for c in df_log.columns]
        energy_col = "Potential Energy (kJ/mole)"
        if energy_col not in df_log.columns:
            return {"complex_id": complex_id, "error": "Energy column missing"}
        energies = df_log[energy_col].values.astype(float)
        
        valid_len = min(dihedrals.shape[0], len(energies))
        if valid_len < cfg["window_frames"]:
            return {"complex_id": complex_id, "error": f"Too few frames: {valid_len}"}
        dihedrals, energies = dihedrals[:valid_len], energies[:valid_len]
        
        H_mean = np.mean(energies)
        delta_H = energies - H_mean
        
        win, step = cfg["window_frames"], cfg["step_frames"]
        SA_vals, H_vals = [], []
        max_t = valid_len - win + 1
        ns_use = cfg["n_states"]
        
        for t in range(0, max_t, step):
            frame_slice = dihedrals[t:t+win]
            energy_slice = delta_H[t:t+win]
            if frame_slice.shape[0] < win or len(energy_slice) == 0: continue
            try:
                sa, debug = calc_entropy_window(frame_slice, ns_use, cfg["lzma_preset"])
                if "error" in debug: continue
                if np.isfinite(sa):
                    SA_vals.append(sa)
                    H_vals.append(float(np.mean(energy_slice)))
            except: continue
                
        if len(SA_vals) < 5:
            return {"complex_id": complex_id, "error": f"Insufficient windows: {len(SA_vals)}"}
            
        df_win = pd.DataFrame({"SA_kB": SA_vals, "H_kJmol": H_vals})
        
        plot_path = os.path.join(cdir, "SA_dH_3D.jpg")
        try:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            valid = df_win["SA_kB"].notna() & df_win["H_kJmol"].notna() & np.isfinite(df_win["SA_kB"])
            sa_d, h_d = df_win.loc[valid, "SA_kB"].values.astype(float), df_win.loc[valid, "H_kJmol"].values.astype(float)
            if len(sa_d) >= 20:
                hist, xe, ye = np.histogram2d(sa_d, h_d, bins=20)
                xc = (xe[:-1]+xe[1:])/2; yc = (ye[:-1]+ye[1:])/2
                X, Y = np.meshgrid(xc, yc, indexing='ij')
                surf = ax.plot_surface(X, Y, hist, cmap='viridis', edgecolor='none', alpha=0.9)
                fig.colorbar(surf, ax=ax, shrink=0.5, label='Occurrence')
            else:
                ax.text2D(0.5, 0.5, f"Low data (n={len(sa_d)})", transform=ax.transAxes)
            ax.set_xlabel(r"$S_A / k_B$"); ax.set_ylabel(r"$\Delta H$ (kJ/mol)")
            ax.set_zlabel("Occurrence"); ax.set_title(complex_id, pad=10)
            ax.view_init(elev=25, azim=45); plt.tight_layout()
            plt.savefig(plot_path, dpi=900, format="jpeg", bbox_inches='tight'); plt.close()
        except Exception:
            pass
            
        # 🔍 Глобальная энтропия + сканирование ns (если включено)
        if cfg.get("scan_ns", False):
            opt_ns, S_global = find_optimal_ns(dihedrals, cfg["ns_range"], cfg["lzma_preset"])
        else:
            opt_ns = cfg["n_states"]
            S_global, _ = calc_entropy_window(dihedrals, cfg["n_states"], cfg["lzma_preset"])
            
        # 📊 Расчёт конформационной свободной энергии: G ≈ H - T·S
        kB_kJ = 0.008314462618  # kJ / (mol·K)
        G_conf = H_mean - cfg["temp_K"] * S_global * kB_kJ
        
        return {
            "complex_id": complex_id, "pep_chain": pep_chain,
            "SA_global_kB": round(float(S_global), 3),
            "SA_window_mean": round(float(df_win["SA_kB"].mean()), 3),
            "H_abs_kJmol": round(float(H_mean), 2),
            "optimal_ns": opt_ns,
            "G_conf_estimate_kJ": round(float(G_conf), 2),
            "plot_path": plot_path, "n_valid_windows": len(SA_vals)
        }
    except Exception as e:
        return {"complex_id": complex_id, "error": f"{type(e).__name__}: {str(e)[:100]}"}

# ================= ОРКЕСТРАЦИЯ =================
def run_pipeline(input_dir: str, output_dir: str, cfg: dict, n_jobs_cpu: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    cfg["output_dir"] = output_dir
    cleanup_log = os.path.join(output_dir, "pipeline_cleanup.log")
    pairs = [(os.path.splitext(os.path.basename(p))[0], p) for p in sorted(glob.glob(os.path.join(input_dir, "*.pdb")))]
    csv_path = os.path.join(output_dir, "entropy_results.csv")
    
    done = set()
    if os.path.exists(csv_path):
        try: done = set(pd.read_csv(csv_path)[pd.read_csv(csv_path)["SA_global_kB"].notna()]["complex_id"].tolist())
        except: pass

    print("\n🔹 Фаза 1: Очистка, выравнивание и запуск MD...")
    for base, p in tqdm(pairs, desc="Prep & MD"):
        if base in done: continue
        cdir = os.path.join(output_dir, base); os.makedirs(cdir, exist_ok=True)
        aligned_pdb = os.path.join(cdir, f"{base}_aligned.pdb")
        dcd, log = os.path.join(cdir, "traj.dcd"), os.path.join(cdir, "md.log")
        if os.path.exists(dcd) and os.path.exists(log) and os.path.getsize(dcd) > 1024**2:
            print(f"  ⏭️ {base}: готово, пропускаю.")
            continue
        try:
            if not (os.path.exists(aligned_pdb) and os.path.getsize(aligned_pdb) > 0):
                print(f"  🔹 {base}: выравнивание...")
                clean_and_align_pdb(p, aligned_pdb)
            else:
                print(f"  ⏭️ {base}: aligned уже есть, пропускаю выравнивание.")

            pep = detect_peptide_chain(aligned_pdb)
            run_implicit_md(aligned_pdb, cdir, cfg, pep)
        except Exception as e:
            msg = f"⚠️ {base}: {type(e).__name__}"
            print(msg); traceback.print_exc()
            with open(cleanup_log, "a", encoding="utf-8") as f: f.write(f"{time.ctime()} | {msg}\n")
            
    print("\n🔹 Фаза 2: Параллельный расчёт энтропии и G...")
    pending = [b for b,_ in pairs if b not in done and os.path.exists(os.path.join(output_dir, b, "traj.dcd"))]
    if pending:
        results = Parallel(n_jobs=n_jobs_cpu, backend="loky")(
            delayed(analyze_single_complex)(b, os.path.join(output_dir, b, f"{b}_aligned.pdb"), cfg) for b in pending
        )
        df_new = pd.DataFrame(results)
        df_old = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        pd.concat([df_old, df_new], ignore_index=True).drop_duplicates("complex_id").to_csv(csv_path, index=False)
        
    print(f"✅ Готово. Таблица: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--platform", default="CPU", choices=["CUDA","CPU","Reference"])
    parser.add_argument("--n_jobs_cpu", type=int, default=4)
    parser.add_argument("--scan_ns", action="store_true", help="Автоматически подобрать оптимальный n_states по минимуму SA")
    args = parser.parse_args()
    CFG["platform"] = args.platform
    CFG["scan_ns"] = args.scan_ns
    run_pipeline(args.input_dir, args.output_dir, CFG, args.n_jobs_cpu)