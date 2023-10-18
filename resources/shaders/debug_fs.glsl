// this is a debug vs for direct output to the screen
#version 330 core

in vec2 vertexUV;

uniform int objID;
uniform sampler2D diffuseTex;
uniform sampler2D normalTex;

out vec4 fragColor;

void main() {
    fragColor = texture(diffuseTex, vertexUV);
}
