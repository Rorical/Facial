import { AudioBufferSource, BufferTarget, CanvasSource, canEncodeAudio, canEncodeVideo, Mp4OutputFormat, Output, QUALITY_HIGH, QUALITY_VERY_HIGH, WebMOutputFormat } from 'mediabunny';
import { MMD_FPS } from '../animation/motion';
import { writeZip, type ZipInput } from './zip';

// Offline export: frames are rendered one by one at a fixed rate (not in real time), so the
// result is smooth whatever the machine manages live.
//   video:  H.264 in MP4 (VP9 in WebM where H.264 encoding is unavailable), optional AAC / Opus audio
//   images: a zip of numbered PNGs

export type ExportFormat = 'video' | 'png';

export interface VideoExportOptions {
  format: ExportFormat;
  /** Inclusive frame range in MMD (30 fps) frames. */
  from: number;
  to: number;
  /** Output frames per second (30 or 60). */
  fps: number;
  /** Decoded audio track, or null for a silent export. Ignored for PNG. */
  audio: AudioBuffer | null;
  onProgress(done: number, total: number): void;
  /** Checked between frames; return true to stop. */
  cancelled(): boolean;
}

export interface VideoExportResult {
  blob: Blob;
  extension: string;
  warnings: string[];
}

/** The part of `audio` that plays during [from, to] (MMD frames). */
function sliceAudio(audio: AudioBuffer, from: number, to: number): AudioBuffer | null {
  const start = Math.round((from / MMD_FPS) * audio.sampleRate);
  const end = Math.min(audio.length, Math.round(((to + 1) / MMD_FPS) * audio.sampleRate));
  if (end <= start) return null;
  const out = new AudioBuffer({ numberOfChannels: audio.numberOfChannels, length: end - start, sampleRate: audio.sampleRate });
  for (let c = 0; c < audio.numberOfChannels; c++) out.copyToChannel(audio.getChannelData(c).subarray(start, end), c);
  return out;
}

/**
 * `renderFrame(frame, dt)` must pose and draw the scene synchronously; the canvas is captured
 * right after it returns. Returns null if cancelled.
 */
export async function exportFrames(
  canvas: HTMLCanvasElement,
  renderFrame: (frame: number, dt: number) => void,
  opts: VideoExportOptions,
): Promise<VideoExportResult | null> {
  const step = MMD_FPS / opts.fps;
  const total = Math.floor((opts.to - opts.from) / step) + 1;
  const dt = 1 / opts.fps;
  const warnings: string[] = [];

  if (opts.format === 'png') {
    const entries: ZipInput[] = [];
    const digits = String(total - 1).length;
    for (let i = 0; i < total; i++) {
      if (opts.cancelled()) return null;
      renderFrame(opts.from + i * step, dt);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('无法读取画面');
      entries.push({ path: `frame_${String(i).padStart(Math.max(digits, 4), '0')}.png`, data: blob, compress: false });
      opts.onProgress(i + 1, total);
    }
    return { blob: await writeZip(entries), extension: 'zip', warnings };
  }

  const h264 = await canEncodeVideo('avc', { width: canvas.width, height: canvas.height });
  if (!h264 && !(await canEncodeVideo('vp9', { width: canvas.width, height: canvas.height }))) {
    throw new Error('这个浏览器不能编码这个尺寸的视频，请改用 PNG 序列或较小的分辨率');
  }
  const output = new Output({ format: h264 ? new Mp4OutputFormat() : new WebMOutputFormat(), target: new BufferTarget() });
  const video = new CanvasSource(canvas, { codec: h264 ? 'avc' : 'vp9', bitrate: QUALITY_VERY_HIGH, keyFrameInterval: 2 });
  output.addVideoTrack(video, { frameRate: opts.fps });

  let audioSource: AudioBufferSource | null = null;
  const audio = opts.audio ? sliceAudio(opts.audio, opts.from, opts.to) : null;
  if (audio) {
    const codec = h264 ? 'aac' : 'opus';
    if (await canEncodeAudio(codec)) {
      audioSource = new AudioBufferSource({ codec, bitrate: QUALITY_HIGH });
      output.addAudioTrack(audioSource);
    } else {
      warnings.push('这个浏览器不能编码音频，导出的视频没有声音');
    }
  }

  await output.start();
  try {
    for (let i = 0; i < total; i++) {
      if (opts.cancelled()) {
        await output.cancel();
        return null;
      }
      renderFrame(opts.from + i * step, dt);
      await video.add(i * dt, dt);
      opts.onProgress(i + 1, total);
    }
    if (audioSource && audio) await audioSource.add(audio);
    await output.finalize();
  } catch (err) {
    await output.cancel().catch(() => {});
    throw err;
  }
  const buffer = output.target.buffer;
  if (!buffer) throw new Error('视频写入失败');
  return { blob: new Blob([buffer], { type: h264 ? 'video/mp4' : 'video/webm' }), extension: h264 ? 'mp4' : 'webm', warnings };
}
