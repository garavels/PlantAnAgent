import { createContext, useContext } from "react";
import type { useWorkspaceState } from "./useWorkspaceState";
export const WorkspaceContext = createContext<ReturnType<
  typeof useWorkspaceState
> | null>(null);
export function useWorkspace() {
  const state = useContext(WorkspaceContext);
  if (!state) throw new Error("Workspace provider missing");
  return state;
}
