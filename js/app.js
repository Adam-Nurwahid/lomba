/**
 * Logika utama aplikasi Tari Saman AI Pose Coach & Rhythm Trainer.
 */

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { playSfx } from "./audio.js";
import {
    unit2D,
    areLandmarksValid,
    extractBodyBones,
    calcBoneErrors,
    calcSimilarity,
    BONE_LABELS,
} from "./pose-math.js";
import {
    drawReferenceTarget,
    drawOverlayOnWebcam,
    drawRhythmNoteLane,
} from "./renderer.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_LOCAL = "./pose_landmarker_lite.task";
const MODEL_CDN = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const $ = sel => document.querySelector(sel);

// Data state aplikasi
let POSES = {};
let SEQ = [];
let TB = [];
let STEPS = [];
let POSE_TIMES = {};
let CURRENT_VIDEO_SRC = "";

const S = {
    mode: "learn",
    run: false,
    step: 0,
    hold: 0,
    sim: 0,
    err: [99, 99, 99, 99],
    ok: false,
    user: null,
    lm: null,
    speed: 0.5,
    notes: [],
    cur: null,
    t0: 0,
    viewMode: "skeleton",
};

let landmarker = null;
let drawingUtils = null;
let lastCamTime = -1;
let tPrev = performance.now();

const cam = $("#cam");
const ov = $("#ov");
const refCanvas = $("#ref");
const refVideo = $("#refVideo");
const laneCanvas = $("#lane");

const fallbackPose = {
    n: "Siap",
    c: "Pose bersiap.",
    b: [[0, 1], [0, 1], [0, 1], [0, 1]],
};

function getTargetPose() {
    if (S.mode === "learn") {
        const key = STEPS[S.step];
        return POSES[key] || fallbackPose;
    }
    if (S.run && S.cur) {
        return POSES[S.cur.p] || fallbackPose;
    }
    return POSES[SEQ[0]] || fallbackPose;
}

function updateStepCounter() {
    if (STEPS.length) {
        $("#stepCounter").textContent = `${S.step + 1} / ${STEPS.length}`;
    }
}

function syncVideoToCurrentPose() {
    if (!refVideo.src || !STEPS.length) return;
    const curPoseId = STEPS[S.step];
    const t = POSE_TIMES[curPoseId] || 0;
    refVideo.currentTime = Math.max(0, t);
}

export function setMode(mode) {
    S.mode = mode;
    S.run = false;
    S.step = 0;
    S.hold = 0;
    S.notes = [];

    document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("on", b.dataset.m === mode));
    if (mode === "test") $("#spd").value = "1";
    else if (mode === "practice") $("#spd").value = "0.5";
    S.speed = +$("#spd").value;

    $("#spd").disabled = mode !== "practice";
    $("#play").hidden = mode === "learn";
    $("#stepNav").style.display = mode === "learn" ? "flex" : "none";
    $("#play").textContent = "▶ Mulai";
    $("#res").textContent = "";

    if (refVideo.src) {
        refVideo.pause();
        syncVideoToCurrentPose();
    }
    updateStepCounter();
}

function finishSession() {
    S.run = false;
    if (refVideo.src) refVideo.pause();

    const c = { PERFECT: 0, GOOD: 0, MISS: 0 };
    S.notes.forEach(n => {
        if (n.j) c[n.j]++;
    });
    const score = Math.round((c.PERFECT * 100 + c.GOOD * 60) / Math.max(1, S.notes.length));
    let bestStr = "";

    if (S.mode === "test") {
        try {
            const key = "saman-score-" + ($("#templatePreset").value || "default");
            const prev = +localStorage.getItem(key) || 0;
            if (score > prev) localStorage.setItem(key, score);
            bestStr = ` · Terbaik: ${Math.max(score, prev)}`;
        } catch { }
    }

    const audioOn = $("#chkAudio").checked;
    playSfx(score >= 65 ? "win" : "good", audioOn);

    const color = score >= 80 ? "var(--ok)" : score >= 60 ? "var(--warn)" : "var(--bad)";
    const label = score >= 85 ? "Sangat Baik 🏆" : score >= 65 ? "Baik 👍" : score >= 40 ? "Cukup" : "Perlu Latihan";
    $("#res").innerHTML = `<span style="color:${color}">Skor ${score}</span> (${label}) · Perfect: ${c.PERFECT}, Good: ${c.GOOD}, Miss: ${c.MISS}${bestStr}`;
    $("#play").textContent = "🔄 Ulangi";
    S.notes = [];
}

function updateSimulation(dt, nw) {
    S.cur = S.run ? S.notes.find(n => !n.j) : null;
    const target = getTargetPose();
    const canEvaluate = S.ok && S.user;

    S.err = canEvaluate ? calcBoneErrors(S.user, target.b) : [99, 99, 99, 99];
    S.sim = canEvaluate ? calcSimilarity(S.err) : 0;

    const audioOn = $("#chkAudio").checked;

    if (S.mode === "learn") {
        const threshold = 0.70;
        S.hold = S.sim >= threshold ? S.hold + dt : Math.max(0, S.hold - 2 * dt);
        if (S.hold >= 0.8) {
            S.hold = 0;
            playSfx("step", audioOn);
            S.step++;
            if (S.step >= STEPS.length) {
                playSfx("win", audioOn);
                setMode("practice");
                $("#res").textContent = "🎉 Selamat! Semua pose sudah dikuasai. Sekarang coba Mode Latihan.";
                return;
            }
            updateStepCounter();
            syncVideoToCurrentPose();
        }
        return;
    }

    if (!S.run) return;

    const now = (nw - S.t0) / 1000;
    const windowSec = 0.6 / S.speed;

    // Sinkronisasi video referensi
    if (refVideo.src && !refVideo.paused) {
        const targetVideoTime = now * S.speed;
        if (Math.abs(refVideo.currentTime - targetVideoTime) > 0.4) {
            refVideo.currentTime = targetVideoTime;
        }
    }

    S.notes.forEach(n => {
        if (n.j) return;
        if (canEvaluate && Math.abs(now - n.T) <= windowSec) {
            const poseItem = POSES[n.p];
            if (poseItem) {
                const errs = calcBoneErrors(S.user, poseItem.b);
                n.best = Math.max(n.best, calcSimilarity(errs));
            }
        }
        if (now > n.T + windowSec) {
            if (n.best >= 0.72) {
                n.j = "PERFECT";
                playSfx("perfect", audioOn);
            } else if (n.best >= 0.48) {
                n.j = "GOOD";
                playSfx("good", audioOn);
            } else {
                n.j = "MISS";
                playSfx("miss", audioOn);
            }
        }
    });

    if (S.notes.every(n => n.j)) {
        finishSession();
    }
}

function renderUI(nw) {
    const target = getTargetPose();
    const now = S.run ? (nw - S.t0) / 1000 : 0;

    // 1. Gambar target di canvas referensi
    drawReferenceTarget(refCanvas, target, S.user, S.ok, S.sim);

    // 2. Gambar overlay webcam
    drawOverlayOnWebcam(ov, S.lm, S.ok, S.err, drawingUtils, PoseLandmarker);

    // 3. Update teks status & kartu informasi
    let titleStr = "";
    let cueStr = target.c || "";

    if (S.mode === "learn") {
        titleStr = `Pose ${S.step + 1} dari ${STEPS.length} · ${target.n}`;
    } else if (S.run) {
        titleStr = now < 3 ? `Bersiap… ${Math.ceil(3 - now)}` : `Gerak ${S.notes.indexOf(S.cur) + 1}/${S.notes.length} · ${target.n}`;
    } else {
        titleStr = S.mode === "test" ? "Mode Uji · Penilaian akurasi pada tempo 1,0×" : "Mode Latihan · Ikuti ketukan ritme";
        cueStr = "Tekan 'Mulai', lalu sesuaikan pose saat kartu melewati garis putih.";
    }

    $("#ttl").textContent = titleStr;
    $("#cue").textContent = cueStr;

    // Bar kecocokan
    const pctVal = Math.round(S.mode === "learn" ? Math.min(100, (S.hold / 0.8) * 100) : S.sim * 100);
    const barElem = $("#bar i");
    barElem.style.width = pctVal + "%";
    barElem.style.backgroundColor = pctVal >= 70 ? "var(--ok)" : pctVal >= 45 ? "var(--warn)" : "var(--bad)";
    $("#bar-pct").textContent = pctVal + "%";

    // Petunjuk perbaikan posisi
    if (!S.lm) {
        $("#hint").innerHTML = `<span style="color:var(--mut)">⚪ Kamera belum mendeteksi tubuh. Pastikan pencahayaan cukup.</span>`;
    } else if (!S.ok) {
        $("#hint").innerHTML = `<span style="color:var(--warn)">⚠️ Posisikan kedua bahu, siku, dan tangan agar masuk ke dalam frame.</span>`;
    } else {
        if (S.sim >= 0.70) {
            $("#hint").innerHTML = `<span style="color:var(--ok);font-weight:600">✓ Pose Selaras! ${S.mode === "learn" ? "Tahan posisi sebentar..." : ""}</span>`;
        } else {
            const maxErrIdx = S.err.indexOf(Math.max(...S.err));
            const maxErrVal = Math.round(S.err[maxErrIdx]);
            $("#hint").innerHTML = `<span style="color:var(--warn)">Sesuaikan posisi: <b>${BONE_LABELS[maxErrIdx]}</b> (selisih ${maxErrVal}°)</span>`;
        }
    }

    // 4. Render Note Lane
    drawRhythmNoteLane(laneCanvas, S, POSES);
}

function onAnimationFrame() {
    requestAnimationFrame(onAnimationFrame);
    const nw = performance.now();
    const dt = Math.min(0.1, (nw - tPrev) / 1000);
    tPrev = nw;

    if (landmarker && cam.readyState >= 2 && cam.currentTime !== lastCamTime) {
        lastCamTime = cam.currentTime;
        const res = landmarker.detectForVideo(cam, nw);
        S.lm = res.poseLandmarks ? res.poseLandmarks[0] : res.landmarks ? res.landmarks[0] : null;

        if (S.lm && areLandmarksValid(S.lm)) {
            S.ok = true;
            const nb = extractBodyBones(S.lm, cam.videoWidth, cam.videoHeight);
            S.user = S.user ? nb.map((v, i) => unit2D([S.user[i][0] * 0.5 + v[0] * 0.5, S.user[i][1] * 0.5 + v[1] * 0.5])) : nb;
        } else {
            S.ok = false;
        }
    }

    updateSimulation(dt, nw);
    renderUI(nw);
}

export async function startCamera() {
    const msg = t => ($("#msg").textContent = t);
    $("#go").disabled = true;
    msg("Meminta izin akses kamera webcam…");

    try {
        cam.srcObject = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
            audio: false,
        });
        await cam.play();
        ov.width = cam.videoWidth;
        ov.height = cam.videoHeight;
        msg("Memuat model Pose AI…");

        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const opts = {
            baseOptions: { modelAssetPath: MODEL_LOCAL, delegate: "GPU" },
            runningMode: "VIDEO",
            numPoses: 1,
        };

        try {
            landmarker = await PoseLandmarker.createFromOptions(vision, opts);
        } catch {
            opts.baseOptions.delegate = "CPU";
            opts.baseOptions.modelAssetPath = MODEL_CDN;
            landmarker = await PoseLandmarker.createFromOptions(vision, opts);
        }

        drawingUtils = new DrawingUtils(ov.getContext("2d"));
        $("#intro").hidden = true;
        msg("");
    } catch (e) {
        msg("Gagal: " + e.message + ". Pastikan webcam terhubung dan beri izin akses.");
        $("#go").disabled = false;
    }
}

export async function loadTemplateFromJson(json, sourceName = "") {
    try {
        const boneKeys = ["L_upper", "L_fore", "R_upper", "R_fore"];
        const loadedPoses = {};

        json.poses.forEach(p => {
            const spreadInfo = p.spreadDeg ? `(variasi ±${p.spreadDeg}°)` : "";
            loadedPoses[p.id] = {
                n: p.id,
                c: `Pose unik ${p.id} ${spreadInfo} dari video tari Saman.`,
                b: boneKeys.map(k => unit2D(p.bones[k].slice(0, 2))),
            };
        });

        const timelineList = json.timeline
            .map(x => ({
                p: x.pose,
                t: x.tStart !== undefined ? x.tStart : x.t,
            }))
            .sort((a, b) => a.t - b.t);

        POSE_TIMES = {};
        timelineList.forEach(item => {
            if (!(item.p in POSE_TIMES)) POSE_TIMES[item.p] = item.t;
        });

        POSES = loadedPoses;
        SEQ = timelineList.map(x => x.p);
        TB = timelineList.map(x => x.t - timelineList[0].t);
        STEPS = [...new Set(SEQ)];

        if (json.source && json.source.video) {
            CURRENT_VIDEO_SRC = `./videos/saman/${json.source.video}`;
            refVideo.src = CURRENT_VIDEO_SRC;
            refVideo.load();
        }

        setMode("learn");
        $("#res").textContent = `Template dimuat: ${STEPS.length} pose unik, ${SEQ.length} ketukan.`;
    } catch (err) {
        $("#res").textContent = "Format template tidak valid: " + err.message;
    }
}

export async function loadPresetTemplate(url) {
    try {
        $("#res").textContent = "Memuat template gerakan...";
        let res = await fetch(url);
        if (!res.ok && !url.startsWith("templates/")) {
            // Coba fallback ke folder templates/
            res = await fetch(`templates/${url}`);
        }
        if (!res.ok) throw new Error(`File tidak ditemukan (${res.status})`);
        const data = await res.json();
        await loadTemplateFromJson(data, url);
    } catch (e) {
        $("#res").textContent = "Gagal memuat preset: " + e.message;
    }
}

function setViewMode(mode) {
    S.viewMode = mode;
    $("#btnViewSkeleton").classList.toggle("active", mode === "skeleton");
    $("#btnViewVideo").classList.toggle("active", mode === "video");
    $("#btnViewBoth").classList.toggle("active", mode === "both");

    if (mode === "skeleton") {
        refCanvas.style.display = "block";
        refVideo.style.display = "none";
        $("#refHeader").textContent = "🎯 Target Pose Acuan (AI Skeleton)";
    } else if (mode === "video") {
        refCanvas.style.display = "none";
        refVideo.style.display = "block";
        $("#refHeader").textContent = "📹 Video Penari Asli (Sinkron)";
    } else if (mode === "both") {
        refCanvas.style.display = "block";
        refVideo.style.display = "block";
        refCanvas.style.opacity = "0.7";
        $("#refHeader").textContent = "✨ Kombinasi Video & Skeleton";
    }
}

// Inisialisasi Event Handlers
export function initApp() {
    $("#go").onclick = startCamera;
    document.querySelectorAll(".tab-btn").forEach(b => (b.onclick = () => setMode(b.dataset.m)));

    $("#spd").onchange = e => {
        S.speed = +e.target.value;
        if (refVideo.src) refVideo.playbackRate = S.speed;
    };

    $("#prevStep").onclick = () => {
        if (S.step > 0) {
            S.step--;
            S.hold = 0;
            updateStepCounter();
            syncVideoToCurrentPose();
        }
    };

    $("#nextStep").onclick = () => {
        if (S.step < STEPS.length - 1) {
            S.step++;
            S.hold = 0;
            updateStepCounter();
            syncVideoToCurrentPose();
        }
    };

    $("#play").onclick = () => {
        if (S.run) {
            S.run = false;
            S.notes = [];
            if (refVideo.src) refVideo.pause();
            $("#play").textContent = "▶ Mulai";
            return;
        }
        if (!landmarker) {
            $("#res").textContent = "Mulai kamera terlebih dahulu.";
            return;
        }

        S.notes = SEQ.map((p, i) => ({
            p,
            T: 3 + TB[i] / S.speed,
            best: 0,
            j: null,
        }));

        S.t0 = performance.now();
        S.run = true;
        $("#res").textContent = "";
        $("#play").textContent = "⏸ Berhenti";

        if (refVideo.src) {
            refVideo.currentTime = 0;
            refVideo.playbackRate = S.speed;
            setTimeout(() => {
                if (S.run) refVideo.play().catch(() => {});
            }, 3000 / S.speed);
        }
    };

    $("#templatePreset").onchange = e => {
        const val = e.target.value;
        if (val === "custom") {
            $("#file").click();
        } else {
            loadPresetTemplate(val);
        }
    };

    $("#file").onchange = async e => {
        if (!e.target.files.length) return;
        try {
            const text = await e.target.files[0].text();
            const json = JSON.parse(text);
            await loadTemplateFromJson(json, e.target.files[0].name);
        } catch (err) {
            $("#res").textContent = "File JSON tidak valid: " + err.message;
        }
    };

    $("#btnViewSkeleton").onclick = () => setViewMode("skeleton");
    $("#btnViewVideo").onclick = () => setViewMode("video");
    $("#btnViewBoth").onclick = () => setViewMode("both");

    // Muat template default awal
    loadPresetTemplate("saman-solo-candidate.template.json");
    requestAnimationFrame(onAnimationFrame);
}

// Jalankan aplikasi saat DOM siap
window.addEventListener("DOMContentLoaded", initApp);
