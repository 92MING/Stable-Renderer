#version 430 core

uniform sampler2D currentColor;   // vec4, rgba
uniform usampler2D currentIDs;      // ivec4 (spriteID, material id, vertexID, 3D pixel index)
uniform sampler2D currentPos;     // vec3, world space position
uniform sampler2D currentNormalDepth; // vec4, (vec3(view space normal), depth)
uniform sampler2D currentNoises; // vec4, latent noise

in vec2 uv;  // screen space UV

out vec4 FragColor;

void main() {
    FragColor = vec4(texture(currentColor, uv).rgb, 1.0);
}
