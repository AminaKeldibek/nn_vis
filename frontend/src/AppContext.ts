import { createContext, useContext } from 'react'
import type { Dim } from './api/client'

export type AppCallbacks = {
  onTensorDimsChange: (nodeId: string, dims: Dim[]) => void
  onModuleParamChange: (nodeId: string, params: Record<string, unknown>) => void
  onGroupToggle: (groupId: string) => void
  onGroupRename: (groupId: string, label: string) => void
  onGroupUngroup: (groupId: string) => void
}

export const AppContext = createContext<AppCallbacks>({
  onTensorDimsChange: () => {},
  onModuleParamChange: () => {},
  onGroupToggle: () => {},
  onGroupRename: () => {},
  onGroupUngroup: () => {},
})

export function useAppCallbacks() {
  return useContext(AppContext)
}
