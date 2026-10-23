// User-supplied WGSL: compiled between a fixed prelude (the declarations the snippet may use)
// and an epilogue (the entry point that calls it). Errors are reported with line numbers of
// the user's own file; on any error the built-in shading stays in place.

export interface ShaderError {
  /** 1-based line in the user's code (0 if the problem is not in it). */
  line: number;
  message: string;
}

export type ShaderKind = 'material' | 'post';

export interface CustomShader {
  name: string;
  code: string;
}

/** What a snippet must define, by kind. */
export const SHADER_ENTRY: Record<ShaderKind, RegExp> = {
  material: /\bfn\s+shade\s*\(/,
  post: /\bfn\s+effect\s*\(/,
};

/** Optional functions a material shader may define; each enables an extra pass. */
export interface MaterialHooks {
  /** `fn outline(s: Surface) -> vec4f`: colours the outline instead of the material's edge colour. */
  outline: boolean;
  /** `fn hairTransparency(s: Surface) -> f32`: eyes and brows show through hair with this opacity (0 = hidden). */
  hairTransparency: boolean;
}

export function detectMaterialHooks(code: string): MaterialHooks {
  return {
    outline: /\bfn\s+outline\s*\(/.test(code),
    hairTransparency: /\bfn\s+hairTransparency\s*\(/.test(code),
  };
}

/** Which kind of shader a file is, from the function it defines. */
export function detectShaderKind(code: string): ShaderKind | null {
  if (SHADER_ENTRY.material.test(code)) return 'material';
  if (SHADER_ENTRY.post.test(code)) return 'post';
  return null;
}

export function formatShaderErrors(errors: ShaderError[]): string[] {
  return errors.map((e) => (e.line > 0 ? `第 ${e.line} 行：${e.message}` : e.message));
}

/**
 * Compile prelude + user code + epilogue. Returns the module, or the errors with line numbers
 * relative to the user's code.
 */
export async function compileCustomShader(
  device: GPUDevice, label: string, prelude: string, code: string, epilogue: string,
): Promise<{ module: GPUShaderModule } | { errors: ShaderError[] }> {
  const head = `${prelude}\n`;
  const offset = head.split('\n').length - 1;
  const userLines = code.split('\n').length;
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label, code: `${head}${code}\n${epilogue}` });
  const info = await module.getCompilationInfo();
  await device.popErrorScope();
  const errors = info.messages.filter((m) => m.type === 'error').map((m) => {
    const line = m.lineNum - offset;
    // A missing or mistyped entry function shows up in the epilogue that calls it.
    return line >= 1 && line <= userLines ? { line, message: m.message } : { line: 0, message: m.message };
  });
  return errors.length ? { errors } : { module };
}

/** Run pipeline creation, turning validation failures into ShaderErrors. */
export async function createPipelineChecked<T>(device: GPUDevice, create: () => Promise<T>): Promise<{ value: T } | { errors: ShaderError[] }> {
  device.pushErrorScope('validation');
  try {
    const value = await create();
    const err = await device.popErrorScope();
    return err ? { errors: [{ line: 0, message: err.message }] } : { value };
  } catch (e) {
    await device.popErrorScope();
    return { errors: [{ line: 0, message: e instanceof Error ? e.message : String(e) }] };
  }
}
