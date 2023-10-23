#version 330 core

layout(location = 0) in vec3 position; // screen space position
layout(location = 1) in vec2 vUV; // screen space UV

out vec2 uv;

void main() {
    gl_Position = vec4(position, 1.0);
    uv = vec2(vUV.x, 1.0 - vUV.y); // flip y axis, since it is data just load from img
}
