import { useMemo } from "react";
import * as THREE from "three";
import type { WeatherView } from "../../types/v2";
import { deriveCloudVisuals } from "./cloudVisuals";

const vertexShader = `varying vec2 p; void main(){p=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`;
const fragmentShader = `
  varying vec2 p; uniform float density; uniform float moment; uniform float shadow; uniform float forecastCoverage;
  float hash(vec2 v){return fract(sin(dot(v,vec2(127.1,311.7)))*43758.5453);}
  float noise(vec2 v){vec2 i=floor(v),f=fract(v); f=f*f*(3.0-2.0*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
  float fbm(vec2 v){float a=0.0,w=0.5;for(int i=0;i<5;i++){a+=w*noise(v);v=v*2.05+3.1;w*=0.5;}return a;}
  void main(){
    vec2 q=p*5.0+vec2(moment*0.018, moment*0.011);
    float n=fbm(q);
    float cloud=smoothstep(0.58-density*0.22,0.76-density*0.14,n);
    if(forecastCoverage>0.5){
      float threshold=0.90-density*0.80;
      cloud=density>=1.0?1.0:smoothstep(threshold-0.08,threshold+0.08,n);
    }
    float edge=smoothstep(0.0,0.15,p.x)*smoothstep(0.0,0.15,p.y)*smoothstep(0.0,0.15,1.0-p.x)*smoothstep(0.0,0.15,1.0-p.y);
    vec3 c=mix(vec3(.79,.84,.78),vec3(.94,.96,.91),n);
    // Coverage follows the forecast; translucency keeps the plant inspectable.
    float opacity=mix(.8,.36,forecastCoverage);
    gl_FragColor=vec4(mix(c,vec3(.07,.10,.08),shadow),cloud*mix(density,1.0,forecastCoverage)*edge*mix(opacity,.38,shadow));
  }
`;
/** Forecast coverage drives amount; shapes and clock-based drift are illustrative. */
export function WeatherClouds({ weather, illustrativeClouds = false }: { weather: WeatherView; illustrativeClouds?: boolean }) {
  const { density, source, castShadow } = deriveCloudVisuals(weather, illustrativeClouds);
  const uniforms = useMemo(
    () => ({
      density: { value: density },
      moment: { value: weather.sample_end_hours },
      shadow: { value: 0 },
      forecastCoverage: { value: source === "forecast" ? 1 : 0 },
    }),
    [density, source, weather.sample_end_hours],
  );
  const shadeUniforms = useMemo(
    () => ({ ...uniforms, shadow: { value: 1 } }),
    [uniforms],
  );
  if (density <= 0) return null;
  return (
    <group>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[-5, 10.5, -8]}
        raycast={() => null}
        renderOrder={3}
      >
        <planeGeometry args={[120, 95]} />
        <shaderMaterial
          uniforms={uniforms}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          transparent
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      {castShadow && <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[-5, 0.06, -8]}
        raycast={() => null}
        renderOrder={1}
      >
        <planeGeometry args={[120, 95]} />
        <shaderMaterial
          uniforms={shadeUniforms}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          transparent
          depthWrite={false}
        />
      </mesh>}
    </group>
  );
}
