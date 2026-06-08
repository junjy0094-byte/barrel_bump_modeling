"""
APDL 수직 extrude된 정사각형+원 mesh를 항아리 형상으로 morphing.

워크플로우:
  1) APDL이 NLIST로 export한 node 좌표 파일 읽기
  2) z별 항아리 반경 R(z) 계산
  3) 각 node에 morphing 변환 적용
  4) NMODIF 명령어 파일 출력
"""

import numpy as np
import re
import sys
from pathlib import Path


# ============================================
# 1. 항아리 프로파일 (이전 답변의 함수 재사용)
# ============================================
def barrel_profile(z, R_bot, R_top, H,
                    bulge_ratio=1.30, z_belly_ratio=0.42,
                    z_bot=0.0):
    """
    z 위치에서 항아리 반경 R(z) 반환.
    범프는 z_bot ~ z_bot+H 범위에 존재.
    """
    R_max = bulge_ratio * max(R_bot, R_top)

    # ---- 내부 계산은 항상 0~H 로컬 좌표로 ----
    z_local = np.atleast_1d(z) - z_bot

    z_belly = z_belly_ratio * H

    P0 = np.array([0.0, R_bot])
    P1 = np.array([z_belly * 0.6, R_max])
    P2 = np.array([z_belly + (H - z_belly) * 0.4, R_max])
    P3 = np.array([H, R_top])

    t = np.linspace(0, 1, 1000)
    bez = ((1-t)**3)[:,None]*P0 + 3*((1-t)**2 * t)[:,None]*P1 \
          + 3*((1-t) * t**2)[:,None]*P2 + (t**3)[:,None]*P3
    z_bez, R_bez = bez[:,0], bez[:,1]

    R_arr = np.interp(z_local, z_bez, R_bez)
    R_arr = np.clip(R_arr, 0, None)
    return R_arr if z_local.size > 1 else float(R_arr[0])


# ============================================
# 2. NLIST 결과 파일 파싱
# ============================================
def parse_nlist(filepath):
    """
    APDL NLIST 출력 파일 파싱.
    형식: 각 데이터 라인이 "node_id  x  y  z  ..."
    헤더/페이지 구분자는 건너뜀.
    """
    nodes = []
    line_re = re.compile(r'^\s*(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)')

    with open(filepath, 'r') as f:
        for line in f:
            m = line_re.match(line)
            if m:
                nid = int(m.group(1))
                x = float(m.group(2))
                y = float(m.group(3))
                z = float(m.group(4))
                nodes.append((nid, x, y, z))

    if not nodes:
        raise RuntimeError(f"No node data parsed from {filepath}")

    arr = np.array(nodes, dtype=[('id', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    print(f"  Parsed {len(arr)} nodes from {filepath}")
    return arr

# ============================================
# 3. Morphing 변환
# ============================================
def morph_nodes(nodes, R0, a_sq, H, R_bot, R_top,
                bulge_ratio=1.30, z_belly_ratio=0.42,
                z_bot=0.0,                          # ← 추가
                tol=1e-9):
    """
    z_bot : 범프 바닥의 절대 z 좌표 (기본 0.0)
            범프는 z_bot ~ z_bot+H 범위에 존재한다고 가정.
    """
    if R0 >= a_sq:
        raise ValueError(f"R0({R0}) >= a_sq({a_sq})")

    x = nodes['x'].copy()
    y = nodes['y'].copy()
    z = nodes['z'].copy()

    # ---- z 범위 sanity check ----
    z_min, z_max = z.min(), z.max()
    if z_min < z_bot - tol or z_max > z_bot + H + tol:
        print(f"  WARNING: node z 범위 [{z_min:.6f}, {z_max:.6f}]가 "
              f"범프 영역 [{z_bot:.6f}, {z_bot+H:.6f}]을 벗어남")

    r_orig = np.hypot(x, y)
    theta  = np.arctan2(y, x)

    # ---- z_bot 전달 ----
    Rz = barrel_profile(z, R_bot, R_top, H,
                         bulge_ratio, z_belly_ratio,
                         z_bot=z_bot)

    R_sq = a_sq / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

    mask_center  = r_orig < tol
    mask_inside  = (~mask_center) & (r_orig <= R0 + tol)
    mask_outside = r_orig > R0 + tol

    r_new = r_orig.copy()
    r_new[mask_inside] = r_orig[mask_inside] * Rz[mask_inside] / R0

    denom = R_sq[mask_outside] - R0
    if denom.size > 0 and np.any(denom < 1e-9):
        print(f"  WARNING: R_sq-R0 매우 작음 (min={denom.min():.3e})")
    r_new[mask_outside] = (Rz[mask_outside]
                            + (r_orig[mask_outside] - R0)
                            * (R_sq[mask_outside] - Rz[mask_outside]) / denom)

    scale = np.divide(r_new, r_orig,
                       out=np.ones_like(r_orig),
                       where=r_orig > tol)

    x_new = x * scale
    y_new = y * scale

    if np.any(np.isnan(x_new)) or np.any(np.isnan(y_new)):
        raise RuntimeError("NaN detected")

    out = nodes.copy()
    out['x'] = x_new
    out['y'] = y_new
    return out

# ============================================
# 4. NMODIF 명령어 출력
# ============================================
def write_nmodif(nodes_new, filepath, chunk=5000):
    """
    NMODIF 명령어를 파일로 출력.
    APDL은 매우 긴 입력 파일도 처리 가능하지만, 가독성을 위해 청크별 진행 메시지 삽입.
    """
    with open(filepath, 'w') as f:
        f.write("! Auto-generated NMODIF commands for barrel morphing\n")
        f.write(f"! Total nodes: {len(nodes_new)}\n")
        f.write("/PREP7\n")
        f.write("CSYS, 0\n\n")

        for i, n in enumerate(nodes_new):
            f.write(f"NMODIF,{n['id']},{n['x']:.10f},{n['y']:.10f},{n['z']:.10f}\n")
            if (i + 1) % chunk == 0:
                f.write(f"/COM, --- {i+1}/{len(nodes_new)} nodes modified ---\n")

        f.write("\nALLSEL,ALL\n")
        f.write("/COM, Morphing complete.\n")

    print(f"  Wrote {len(nodes_new)} NMODIF commands to {filepath}")

# ============================================
# 5. (선택) 시각화로 사전 검증
# ============================================
def preview(nodes_new, R0, a_sq, H, R_bot, R_top,
            bulge_ratio=1.30, z_belly_ratio=0.42,
            z_bot=0.0):                              # ← 추가
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))

    # (a) 항아리 프로파일 (절대 z로 표시)
    ax1 = fig.add_subplot(1, 3, 1)
    zs = np.linspace(z_bot, z_bot + H, 200)
    Rs = barrel_profile(zs, R_bot, R_top, H,
                         bulge_ratio, z_belly_ratio, z_bot=z_bot)
    ax1.plot(Rs, zs, 'b-')
    ax1.plot(-Rs, zs, 'b-')
    ax1.fill_betweenx(zs, -Rs, Rs, alpha=0.2)
    ax1.axvline(R0, color='r', ls=':', label=f'R0={R0}')
    ax1.axvline(-R0, color='r', ls=':')
    ax1.axhline(z_bot, color='gray', ls='--', alpha=0.5)
    ax1.set_xlabel('R'); ax1.set_ylabel('z (absolute)')
    ax1.set_aspect('equal'); ax1.grid(True); ax1.legend()
    ax1.set_title(f'Barrel profile (z_bot={z_bot})')

    # (b) 바닥면
    ax2 = fig.add_subplot(1, 3, 2)
    bot = nodes_new[np.abs(nodes_new['z'] - z_bot) < 1e-6]
    ax2.scatter(bot['x'], bot['y'], s=1, c='b')
    ax2.set_aspect('equal'); ax2.grid(True)
    ax2.set_title(f'Bottom (z={z_bot}), {len(bot)} nodes')

    # (c) belly 단면
    ax3 = fig.add_subplot(1, 3, 3)
    z_belly_abs = z_bot + z_belly_ratio * H
    mid = nodes_new[np.abs(nodes_new['z'] - z_belly_abs) < H/16]
    ax3.scatter(mid['x'], mid['y'], s=1, c='g')
    ax3.set_aspect('equal'); ax3.grid(True)
    ax3.set_title(f'Belly (z≈{z_belly_abs:.4f}), {len(mid)} nodes')

    plt.tight_layout()
    plt.savefig('morph_preview.png', dpi=120)
    plt.show()

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    # R0    = 0.030       # mm
    print(sys.argv[1])
    R0 = float(sys.argv[1])
    print(R0)
    # a_sq  = 0.050
    a_sq = float(sys.argv[2])
    # H     = 0.060
    H = float(sys.argv[3])
    R_bot = R0          # 바닥에서는 원 그대로
    # R_top = 0.025
    R_top = float(sys.argv[4])
    bulge_ratio   = 1.20
    z_belly_ratio = 0.42
    z_bot = float(sys.argv[5])

    here = Path(__file__).parent
    nlist_file = here / "nodes_premesh.txt"
    out_file   = here / "nmodif_cmds.inp"

    print("[1/4] Parsing NLIST output...")
    nodes = parse_nlist(nlist_file)

    if z_bot is None:
        z_bot_auto = nodes['z'].min()
        print(f"  z_bot 자동 감지: {z_bot_auto:.6f}")
        z_bot = z_bot_auto

    print("[2/4] Morphing nodes...")
    nodes_new = morph_nodes(nodes, R0, a_sq, H, R_bot, R_top,
                             bulge_ratio, z_belly_ratio,
                             z_bot=z_bot)              # ← 전달

    print("[3/4] Writing NMODIF commands...")
    write_nmodif(nodes_new, out_file)

    print("[4/4] Preview...")
