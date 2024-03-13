#version 430 core

uniform sampler2D gColor; // color and depth (r, g, b, depth) in g-buffer
uniform sampler2D gPos; // global position in g-buffer
uniform sampler2D gNormal; // global normal in g-buffer
uniform sampler2D gID; // uv and obj-ID in g-buffer
uniform sampler2D gNoise; // noise texture
uniform sampler2D gDepth; // depth texture

in vec2 uv;  // screen space UV

out vec4 FragColor;

void main() {
    FragColor = vec4(texture(gColor, uv).rgb, 1.0);
}
