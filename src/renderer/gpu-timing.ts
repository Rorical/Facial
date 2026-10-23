// GPU time per pass, from timestamp queries (needs the `timestamp-query` feature). Results
// arrive a few frames late; `times` holds a smoothed value per label in milliseconds.

const MAX_PASSES = 24;

export class GpuTiming {
  readonly times = new Map<string, number>();
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffers: GPUBuffer[] = [];
  private labels: string[] = [];
  private busy = 0;

  constructor(private readonly device: GPUDevice) {
    this.querySet = device.createQuerySet({ type: 'timestamp', count: MAX_PASSES * 2 });
    this.resolveBuffer = device.createBuffer({ size: MAX_PASSES * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  }

  static create(device: GPUDevice): GpuTiming | null {
    return device.features.has('timestamp-query') ? new GpuTiming(device) : null;
  }

  beginFrame(): void {
    this.labels = [];
  }

  /** `timestampWrites` for a pass; passes sharing a label are summed. */
  pass(label: string): GPUComputePassTimestampWrites | undefined {
    const i = this.labels.length;
    if (i >= MAX_PASSES) return undefined;
    this.labels.push(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 };
  }

  /** Call before finishing the frame's encoder. */
  resolve(encoder: GPUCommandEncoder): (() => void) | null {
    const labels = this.labels;
    if (labels.length === 0 || this.busy >= 3) return null;
    const bytes = labels.length * 16;
    encoder.resolveQuerySet(this.querySet, 0, labels.length * 2, this.resolveBuffer, 0);
    const read = this.readBuffers.pop() ?? this.device.createBuffer({ size: MAX_PASSES * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, read, 0, bytes);
    // Returned callback starts the readback; call it after submitting.
    return () => {
      this.busy++;
      read.mapAsync(GPUMapMode.READ, 0, bytes).then(() => {
        const t = new BigInt64Array(read.getMappedRange(0, bytes));
        const sums = new Map<string, number>();
        labels.forEach((label, i) => {
          const ns = Number(t[i * 2 + 1] - t[i * 2]);
          // Timestamps can be reset between passes on some drivers; drop nonsense.
          if (ns >= 0 && ns < 1e9) sums.set(label, (sums.get(label) ?? 0) + ns / 1e6);
        });
        read.unmap();
        for (const [label, ms] of sums) this.times.set(label, (this.times.get(label) ?? ms) * 0.9 + ms * 0.1);
        for (const label of this.times.keys()) if (!sums.has(label)) this.times.delete(label);
        this.readBuffers.push(read);
        this.busy--;
      }, () => this.busy--);
    };
  }
}
