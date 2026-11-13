/**
 * Operasi vektor dan matematika pencocokan pose Shoulder-Axis Body Frame.
 */

export const BONE_LABELS = [
    "Lengan Atas Kiri",
    "Lengan Bawah Kiri",
    "Lengan Atas Kanan",
    "Lengan Bawah Kanan",
];

export const KEY_INDICES = [11, 12, 13, 14, 15, 16];

/** Normalisasi vektor 2D menjadi vektor satuan */
export function unit2D(v) {
    const norm = Math.hypot(v[0], v[1]) || 1.0;
    return [v[0] / norm, v[1] / norm];
}

/** Periksa apakah seluruh landmark lengan utama terdeteksi dengan baik */
export function areLandmarksValid(lm) {
    return KEY_INDICES.every(
        i => lm[i] && lm[i].visibility > 0.5 && lm[i].x > 0 && lm[i].x < 1 && lm[i].y > 0 && lm[i].y < 1
    );
}

/**
 * Ekstraksi vektor 4 tulang lengan dalam kerangka sumbu bahu (Shoulder-Axis Body Frame):
 * - ex: Garis bahu kanan ke bahu kiri
 * - ey: Garis tegak lurus mengarah ke bawah
 */
export function extractBodyBones(lm, width, height) {
    const pt = i => [lm[i].x * width, lm[i].y * height];

    const shoulderDiff = [pt(11)[0] - pt(12)[0], pt(11)[1] - pt(12)[1]];
    const ex = unit2D(shoulderDiff);
    const ey = [-ex[1], ex[0]];

    const getBone = (a, b) => {
        const d = [pt(b)[0] - pt(a)[0], pt(b)[1] - pt(a)[1]];
        return unit2D([d[0] * ex[0] + d[1] * ex[1], d[0] * ey[0] + d[1] * ey[1]]);
    };

    return [
        getBone(11, 13), // L_upper
        getBone(13, 15), // L_fore
        getBone(12, 14), // R_upper
        getBone(14, 16), // R_fore
    ];
}

/**
 * Hitung selisih sudut (dalam derajat) untuk setiap tulang terhadap pose acuan.
 */
export function calcBoneErrors(userBones, targetBones) {
    return userBones.map((u, i) => {
        const t = targetBones[i];
        const dot = Math.max(-1.0, Math.min(1.0, u[0] * t[0] + u[1] * t[1]));
        return Math.acos(dot) * (180.0 / Math.PI);
    });
}

/**
 * Hitung skor kecocokan keseluruhan (0.0 s/d 1.0) menggunakan Gaussian kernel.
 */
export function calcSimilarity(errors, sigma = 25.0) {
    const sum = errors.reduce((acc, err) => acc + Math.exp(-0.5 * (err / sigma) ** 2), 0);
    return sum / errors.length;
}
