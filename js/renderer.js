/**
 * Modul perenderan canvas 2D: Stickman Tari Saman, landmark overlay, dan rhythm note lane.
 */

export function getCanvasContext(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [ctx, w, h];
}

/**
 * Gambar stickman pose Tari Saman (duduk di lantai/paha).
 */
export function drawStickman(ctx, bones, x, y, s, col, lw = 3) {
    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const L = [x - 0.5 * s, y];
    const R = [x + 0.5 * s, y];

    const arm = (sh, u, f) => {
        const e = [sh[0] - u[0] * 0.8 * s, sh[1] + u[1] * 0.8 * s];
        return [sh, e, [e[0] - f[0] * 0.7 * s, e[1] + f[1] * 0.7 * s]];
    };

    const line = pts => {
        ctx.beginPath();
        pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
        ctx.stroke();
    };

    // Garis bahu & lengan kiri/kanan
    line([L, R]);
    line(arm(L, bones[0], bones[1]));
    line(arm(R, bones[2], bones[3]));

    // Torso duduk & paha horizontal khas Tari Saman
    line([L, [x - 0.35 * s, y + 1.65 * s], [x + 0.35 * s, y + 1.65 * s], R]);
    line([[x, y], [x, y - 0.25 * s]]);

    // Kepala
    ctx.beginPath();
    ctx.arc(x, y - 0.55 * s, 0.3 * s, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
}

/**
 * Render visualisasi stickman target acuan & bayangan posisi pengguna di canvas referensi.
 */
export function drawReferenceTarget(canvas, targetPose, userPose, isUserOk, similarity) {
    const [c, w, h] = getCanvasContext(canvas);
    c.fillStyle = "#141722";
    c.fillRect(0, 0, w, h);

    // Garis grid lembut
    c.strokeStyle = "rgba(255, 255, 255, 0.04)";
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(w / 2, 0); c.lineTo(w / 2, h);
    c.moveTo(0, h * 0.4); c.lineTo(w, h * 0.4);
    c.stroke();

    // Stickman pose target acuan (Cyan)
    drawStickman(c, targetPose.b, w / 2, h * 0.4, h * 0.25, "#4fc3f7", Math.max(3.5, h / 65));

    // Bayangan pose pengguna real-time (Hijau jika selaras, oranye jika masih berbeda)
    if (isUserOk && userPose) {
        const overlayColor = similarity >= 0.70 ? "rgba(46, 204, 113, 0.65)" : "rgba(243, 156, 18, 0.55)";
        drawStickman(c, userPose, w / 2, h * 0.4, h * 0.25, overlayColor, Math.max(3, h / 70));
    }
}

/**
 * Render garis tulang di atas webcam dengan indikator warna sesuai akurasi.
 */
export function drawOverlayOnWebcam(canvas, lm, isOk, errors, drawingUtils, poseLandmarkerClass) {
    const o = canvas.getContext("2d");
    o.clearRect(0, 0, canvas.width, canvas.height);

    if (lm && drawingUtils) {
        drawingUtils.drawConnectors(lm, poseLandmarkerClass.POSE_CONNECTIONS, {
            color: "rgba(255, 255, 255, 0.35)",
            lineWidth: 2,
        });
        drawingUtils.drawLandmarks(lm, { color: "#ffffff", radius: 2 });

        if (isOk && errors) {
            o.lineWidth = 6;
            o.lineCap = "round";
            const pairs = [[11, 13], [13, 15], [12, 14], [14, 16]];
            pairs.forEach(([a, b], i) => {
                const err = errors[i];
                o.strokeStyle = err < 25 ? "#2ecc71" : err < 45 ? "#f39c12" : "#e74c3c";
                o.beginPath();
                o.moveTo(lm[a].x * canvas.width, lm[a].y * canvas.height);
                o.lineTo(lm[b].x * canvas.width, lm[b].y * canvas.height);
                o.stroke();
            });
        }
    }
}

/**
 * Render lajur kartu ketukan ritmis (Rhythm Note Lane).
 */
export function drawRhythmNoteLane(canvas, state, poses) {
    const [c, w, h] = getCanvasContext(canvas);
    c.fillStyle = "#11131a";
    c.fillRect(0, 0, w, h);

    const hx = w * 0.18;
    const pps = 180 * state.speed;
    const now = state.run ? (performance.now() - state.t0) / 1000 : 0;

    // Judgement hit zone (Garis putih & area target)
    c.fillStyle = "rgba(255, 255, 255, 0.06)";
    c.fillRect(hx - 36, 0, 72, h);
    c.fillStyle = "#4f8cff";
    c.fillRect(hx - 1, 0, 3, h);

    if (!state.notes.length) {
        c.fillStyle = "#8b92a5";
        c.font = "14px system-ui";
        c.textAlign = "center";
        c.fillText(
            state.mode === "learn"
                ? "Mode Belajar aktif: kuasai setiap pose secara bertahap. Note Lane aktif di Mode Latihan & Uji."
                : "Tekan tombol 'Mulai' di bawah untuk memulai ketukan tari.",
            w / 2,
            h / 2 + 4
        );
        return;
    }

    // Gambar kartu-kartu notasi bergerak
    state.notes.forEach(n => {
        const x = hx + (n.T - now) * pps;
        if (x < -60 || x > w + 60) return;

        const poseItem = poses[n.p];
        if (!poseItem) return;

        const col = n.j
            ? { PERFECT: "#2ecc71", GOOD: "#f39c12", MISS: "#e74c3c" }[n.j]
            : "#ffffff";

        c.fillStyle = col;
        c.beginPath();
        c.roundRect(x - 30, 8, 60, h - 16, 10);
        c.fill();

        // Mini stickman di tengah kartu
        drawStickman(c, poseItem.b, x, h * 0.42, 11, "#000000", 2.2);

        c.fillStyle = "#000000";
        c.font = "bold 11px system-ui";
        c.textAlign = "center";
        c.fillText(n.j || poseItem.n, x, h - 14);
    });
}
