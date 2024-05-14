// default post processing FS
#version 430 core

in vec2 uv;
uniform sampler2D screenTexture;

uniform int enableGammaCorrection;
uniform int enableHDR;
uniform float gamma;
uniform float exposure;
uniform float saturation;
uniform float brightness;
uniform float contrast;

// flags
uniform int diffusion_disable;  // whether diffusion process is enabled.
uniform int is_baking;  // whether baking process is enabled.

out vec4 fragColor;

void main() {
    
    vec4 color;
    color = texture(screenTexture, uv); // no need to flip
    
    vec3 rgb = color.rgb;
    if (enableGammaCorrection == 1) {
        rgb = pow(rgb, vec3(1.0 / gamma)); // gamma correction, default 1.0 (no change)
    }
    rgb = rgb * exposure; // exposure, default 1.0 (no change)
    rgb = mix(vec3(0.5), rgb, saturation); // saturation, default 1.0 (no change)
    rgb = rgb * brightness; // brightness, default 1.0 (no change)
    rgb = (rgb - vec3(0.5)) * contrast + vec3(0.5); // contrast, default 1.0 (no change)

    if (enableHDR == 1) {
        rgb = rgb / (rgb + vec3(1.0)); // HDR
    }
    fragColor = vec4(rgb, color.a);
}
