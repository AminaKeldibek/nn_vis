const BASE = 'http://localhost:8000/api/v1'

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (r.status === 204) return undefined as T
  const data = await r.json()
  if (!r.ok) {
    const errs = data?.detail?.errors
    const msg = errs
      ? errs.map((e: { message: string }) => e.message).join(', ')
      : data?.detail ?? 'Request failed'
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return data
}

export type Dim = number | string
export type TensorShape = { dims: Dim[] }

export type TensorData = { id: string; shape: TensorShape; dtype: string }
export type ModuleData = { id: string; type: string; params: Record<string, unknown> }
export type ConnectionData = { id: string; source_id: string; target_id: string; output_tensor_id?: string; output_tensor_ids?: string[] }

export type TensorResponse = { tensor: TensorData; updated_tensors: TensorData[]; validation: ValidationResult }
export type ModuleResponse = { module: ModuleData; updated_tensors: TensorData[]; validation: ValidationResult }
export type ConnectionResponse = { connection: ConnectionData; output_tensor?: TensorData; output_tensors?: TensorData[]; validation: ValidationResult }
export type ValidationResult = { ok: boolean; errors: { code: string; message: string }[] }
export type GroupData = { id: string; name: string; member_ids: string[] }
export type GroupResponse = { group: GroupData; validation: ValidationResult }

export const api = {
  createTensor: (dims: Dim[], dtype = 'float32') =>
    request<TensorResponse>('POST', '/tensors', { shape: { dims }, dtype }),
  patchTensor: (id: string, shape: { dims: Dim[] }) =>
    request<TensorResponse>('PATCH', `/tensors/${id}`, { shape }),
  deleteTensor: (id: string) => request<void>('DELETE', `/tensors/${id}`),

  createModule: (type: string, params: Record<string, unknown>) =>
    request<ModuleResponse>('POST', '/modules', { type, params }),
  patchModule: (id: string, params: Record<string, unknown>) =>
    request<ModuleResponse>('PATCH', `/modules/${id}`, { params }),
  deleteModule: (id: string) => request<void>('DELETE', `/modules/${id}`),

  createConnection: (source_id: string, target_id: string, target_handle?: string | null) =>
    request<ConnectionResponse>('POST', '/connections', { source_id, target_id, target_handle: target_handle ?? null }),
  deleteConnection: (id: string) => request<void>('DELETE', `/connections/${id}`),

  createGroup: (name: string, member_ids: string[]) =>
    request<GroupResponse>('POST', '/groups', { name, member_ids }),
  patchGroup: (id: string, body: { name?: string; member_ids?: string[] }) =>
    request<GroupResponse>('PATCH', `/groups/${id}`, body),
  deleteGroup: (id: string) => request<void>('DELETE', `/groups/${id}`),
}
