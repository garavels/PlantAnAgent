import type { ReactNode } from "react";
import { WorkspaceContext } from "./context";
import { useWorkspaceState } from "./useWorkspaceState";
export function WorkspaceProvider({ children }: { children: ReactNode }) {
  return (
    <WorkspaceContext.Provider value={useWorkspaceState()}>
      {children}
    </WorkspaceContext.Provider>
  );
}
