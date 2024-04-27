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

// whether StableDiffusion is enabled.
// if enabled, img will have been flipped vertically.
uniform int usingSD;

out vec4 fragColor;

void main() {

    vec4 color;

    if (usingSD == 1) {
        color = texture(screenTexture, uv); // no need to flip
    } else {
        vec2 uv_flip = vec2(uv.x, 1.0 - uv.y);
        color = texture(screenTexture, uv_flip);
    }

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
