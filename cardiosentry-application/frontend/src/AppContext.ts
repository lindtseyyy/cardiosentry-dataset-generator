import { createContext, useContext } from "react";
import type { AppConfig, HealthResponse, ModelsResponse } from "./api/types";

export interface AppData {
  config: AppConfig | null;
  health: HealthResponse | null;
  models: ModelsResponse | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

export const AppContext = createContext<AppData>({
  config: null,
  health: null,
  models: null,
  loading: true,
  error: null,
  reload: () => {},
});

export function useApp(): AppData {
  return useContext(AppContext);
}
