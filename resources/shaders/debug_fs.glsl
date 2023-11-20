// this is a debug vs for direct output to the screen
#version 330 core

in vec2 vertexUV;

uniform int objID;

uniform sampler2D diffuseTex;
uniform int hasDiffuseTex;
uniform sampler2D normalTex;
uniform int hasNormalTex;

uniform int grayMode; // turn color to gray
uniform int pinkMode; // turn color to all pink. Used when missing texture, etc.
uniform int whiteMode; // turn color to all white. Used when missing texture, etc.

out vec4 fragColor;

void main() {
    if (grayMode == 1) {
        vec4 color = texture(diffuseTex, vertexUV);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        fragColor = vec4(gray, gray, gray, 1.0);
    }
    else if (whiteMode == 1) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    else if (pinkMode == 1) {
        fragColor = vec4(1.0, 0.0, 1.0, 1.0);
    }
    else{
        fragColor = texture(diffuseTex, vertexUV);
    }
}
