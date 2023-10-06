#version 430
//fragment shader of Point Light

in layout (location = 0) vec3 position;
in layout (location = 1) vec2 uv;
in layout (location = 2) vec3 normal;

vec4 worldPos; //world pos of model vertex

uniform int hasDiffuseTex;
uniform sampler2D diffuseTex;
uniform vec3 lightPos;
uniform mat4 modelMatrix;
uniform mat4 lightVP;
uniform float farPlane;

void main()
{
    if (hasDiffuseTex ==1){
        if (texture(diffuseTex, uv).a < 0.4) discard;
    }
    //覆蓋gl_Position的深度
    gl_FragDepth = length(worldPos.xyz - lightPos) / farPlane;
}