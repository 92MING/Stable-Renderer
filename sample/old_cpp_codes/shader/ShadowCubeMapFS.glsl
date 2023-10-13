#version 430

uniform vec3 lightPos;
uniform float farPlane;
in vec4 fragPos;

uniform int checkAlpha;
uniform sampler2D colorMap;
in vec2 texCoords;

void main()
{
    if (checkAlpha == 1){
        if (texture(colorMap, texCoords).r >= 0.25)
			discard;
    }
    gl_FragDepth = length(fragPos.xyz - lightPos) / farPlane;
}