// @mkkellogg/gaussian-splats-3d ships no type definitions. Only the surface the
// Preview page touches is declared here — keep it minimal and honest.
declare module "@mkkellogg/gaussian-splats-3d" {
  export const SceneFormat: { Splat: 0; KSplat: 1; Ply: 2; Spz: 3 };

  export interface ViewerOptions {
    rootElement?: HTMLElement;
    cameraUp?: [number, number, number];
    initialCameraPosition?: [number, number, number];
    initialCameraLookAt?: [number, number, number];
    sharedMemoryForWorkers?: boolean;
    selfDrivenMode?: boolean;
    useBuiltInControls?: boolean;
    sphericalHarmonicsDegree?: number;
  }

  export interface AddSplatSceneOptions {
    format?: number;
    showLoadingUI?: boolean;
    progressiveLoad?: boolean;
    splatAlphaRemovalThreshold?: number;
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    addSplatScene(path: string, options?: AddSplatSceneOptions): Promise<void>;
    start(): void;
    dispose(): Promise<void>;
  }
}
