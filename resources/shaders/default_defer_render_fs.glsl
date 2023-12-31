#version 330 core

uniform sampler2D gColor_and_depth; // color and depth (r, g, b, depth) in g-buffer
uniform sampler2D gPos; // global position in g-buffer
uniform sampler2D gNormal; // global normal in g-buffer
uniform sampler2D g_UV_and_ID; // uv and obj-ID in g-buffer

in vec2 uv;  // screen space UV

out vec4 FragColor;

void main() {
    FragColor = vec4(texture(gColor_and_depth, uv).rgb, 1.0);
}
