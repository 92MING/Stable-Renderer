// this is a debug vs for direct output to the screen
#version 430 core

in vec2 vertexUV;

uniform sampler2D diffuseTex;
uniform int hasDiffuseTex;
uniform sampler2D normalTex;
uniform int hasNormalTex;

uniform int mode; // turn color to gray
//  modes:
//  0: pink(default) - turn missing textures to pink
//  1: white - turn all missing textures to white
//  2: gray - turn all colors to gray(satuation = 0)
out vec4 fragColor;

void main() {
    if (mode == 2) { // gray
        vec4 color = texture(diffuseTex, vertexUV);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        fragColor = vec4(gray, gray, gray, 1.0);
    }
    else if (mode == 1) { // white
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    else if (mode == 0) { // pink
        fragColor = vec4(1.0, 0.0, 1.0, 1.0);
    }
    else{
        fragColor = texture(diffuseTex, vertexUV);
    }
}
