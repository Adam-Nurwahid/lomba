/**
 * Web Audio API Synthesizer untuk efek suara interaktif.
 */

let audioCtx = null;

export function getAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
        audioCtx.resume();
    }
    return audioCtx;
}

export function playSfx(type, enabled = true) {
    if (!enabled) return;
    try {
        const ctx = getAudioContext();
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        if (type === "step") {
            // Suara kemajuan langkah di mode Belajar
            osc.type = "sine";
            osc.frequency.setValueAtTime(523.25, now);
            osc.frequency.setValueAtTime(659.25, now + 0.08);
            gain.gain.setValueAtTime(0.18, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
            osc.start(now);
            osc.stop(now + 0.22);
        } else if (type === "perfect") {
            // Suara ketukan sempurna (Perfect chime)
            osc.type = "triangle";
            osc.frequency.setValueAtTime(659.25, now);
            osc.frequency.setValueAtTime(987.77, now + 0.07);
            gain.gain.setValueAtTime(0.22, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
            osc.start(now);
            osc.stop(now + 0.25);
        } else if (type === "good") {
            // Suara ketukan baik (Good tap)
            osc.type = "sine";
            osc.frequency.setValueAtTime(440, now);
            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
            osc.start(now);
            osc.stop(now + 0.15);
        } else if (type === "miss") {
            // Suara ketukan meleset (Miss thud)
            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(140, now);
            osc.frequency.exponentialRampToValueAtTime(70, now + 0.16);
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.16);
            osc.start(now);
            osc.stop(now + 0.16);
        } else if (type === "win") {
            // Melodi kemenangan / tamat sesi
            const notes = [523.25, 659.25, 783.99, 1046.50];
            notes.forEach((f, i) => {
                const o = ctx.createOscillator();
                const g = ctx.createGain();
                o.connect(g);
                g.connect(ctx.destination);
                o.type = "triangle";
                o.frequency.setValueAtTime(f, now + i * 0.09);
                g.gain.setValueAtTime(0.2, now + i * 0.09);
                g.gain.exponentialRampToValueAtTime(0.001, now + i * 0.09 + 0.25);
                o.start(now + i * 0.09);
                o.stop(now + i * 0.09 + 0.25);
            });
        }
    } catch (e) {
        // Redam kegagalan audio jika browser membatasi autoplay
    }
}
