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
                r_bot_rev=None,                     # ← 추가
                tol=1e-9):
    """
    z_bot : 범프 바닥의 절대 z 좌표 (기본 0.0)
            범프는 z_bot ~ z_bot+H 범위에 존재한다고 가정.
    r_bot_rev : z_bot 이하(base) 영역의 목표 원 반경.
            - 원래 반경 R0 인 원통 노드를 r_bot_rev 로 균일 수축시키고,
              그 내·외부 노드도 동일한 morphing 규칙으로 함께 이동한다
              (z_bot 이하 모든 단면에 동일 적용 → 원통이 깨지지 않음).
            - barrel 바닥(z=z_bot) 반경도 r_bot_rev 가 되어 base 영역과
              연속으로 이어진다(단차 제거).
            - None 이면 R_bot 으로 두어 기존 거동을 그대로 유지.

    NOTE: base 영역(z<=z_bot)과 barrel 영역(z>z_bot)은 노드 z가 겹치지 않는
          서로소 집합이므로, "base 먼저 → barrel 나중" 변환을 아래의 한 번의
          piecewise Rz 계산으로 동일하게 구현한다.
    """
    if R0 >= a_sq:
        raise ValueError(f"R0({R0}) >= a_sq({a_sq})")
    if r_bot_rev is None:
        r_bot_rev = R_bot

    x = nodes['x'].copy()
    y = nodes['y'].copy()
    z = nodes['z'].copy()

    # ---- z 범위 sanity check ----
    # z_bot 이하는 의도적으로 base 수축이 적용되는 영역이므로 경고하지 않는다.
    z_min, z_max = z.min(), z.max()
    if z_max > z_bot + H + tol:
        print(f"  INFO: z_bot+H({z_bot+H:.6f}) 위쪽 노드 존재 "
              f"(z_max={z_max:.6f}) → R_top 반경으로 처리됨")

    r_orig = np.hypot(x, y)
    theta  = np.arctan2(y, x)

    # ---- barrel 바닥을 r_bot_rev 로 설정 → base 영역과 연속 ----
    Rz = barrel_profile(z, r_bot_rev, R_top, H,
                         bulge_ratio, z_belly_ratio,
                         z_bot=z_bot)
    # ---- z_bot 이하 base: 원통형을 r_bot_rev 로 균일 수축 ----
    base_mask = z <= z_bot + tol
    Rz = np.where(base_mask, r_bot_rev, Rz)

    # ---- 진단: base 수축 영역으로 잡힌 노드 수/범위 출력 ----
    n_base = int(np.count_nonzero(base_mask))
    if n_base:
        zb = z[base_mask]
        print(f"  base 수축영역(z<=z_bot={z_bot:.6f}): {n_base}/{len(z)} nodes, "
              f"z범위 [{zb.min():.6f}, {zb.max():.6f}] -> R={r_bot_rev}")
    else:
        print(f"  WARNING: base 수축영역(z<=z_bot={z_bot:.6f})에 해당하는 노드가 없음! "
              f"(node z범위 [{z_min:.6f}, {z_max:.6f}]) → z_bot 값/좌표계 확인 필요")

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
            z_bot=0.0, r_bot_rev=None):              # ← 추가
    import matplotlib.pyplot as plt
    if r_bot_rev is None:
        r_bot_rev = R_bot

    fig = plt.figure(figsize=(12, 5))

    # (a) 항아리 프로파일 (절대 z로 표시, z_bot 아래 base 수축 영역 포함)
    ax1 = fig.add_subplot(1, 3, 1)
    z_low = z_bot - 0.3 * H
    zs = np.linspace(z_low, z_bot + H, 260)
    Rs = barrel_profile(zs, r_bot_rev, R_top, H,
                         bulge_ratio, z_belly_ratio, z_bot=z_bot)
    Rs = np.where(zs <= z_bot, r_bot_rev, Rs)   # base 영역은 균일 수축
    ax1.plot(Rs, zs, 'b-')
    ax1.plot(-Rs, zs, 'b-')
    ax1.fill_betweenx(zs, -Rs, Rs, alpha=0.2)
    ax1.axvline(R0, color='r', ls=':', label=f'R0={R0}')
    ax1.axvline(-R0, color='r', ls=':')
    ax1.axvline(r_bot_rev, color='m', ls='-.', label=f'r_bot_rev={r_bot_rev}')
    ax1.axvline(-r_bot_rev, color='m', ls='-.')
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
    # r_bot_rev: z_bot 이하 base 영역의 목표 원 반경 (선택).
    #   미지정 시 R0 → base 수축 없음(기존 거동). 배럴 바닥도 r_bot_rev 로 시작.
    r_bot_rev = float(sys.argv[6]) if len(sys.argv) > 6 else R0

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
                             z_bot=z_bot,
                             r_bot_rev=r_bot_rev)      # ← 전달

    print("[3/4] Writing NMODIF commands...")
    write_nmodif(nodes_new, out_file)

    print("[4/4] Preview...")
