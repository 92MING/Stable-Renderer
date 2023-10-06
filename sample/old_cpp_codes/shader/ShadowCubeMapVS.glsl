#version 430
//vertex shader for shadow cube map (point light)

in layout (location = 0) vec3 position;
in layout (location = 1) vec2 uv;
in layout (location = 2) vec3 normal;

uniform int hasDiffuseTex;
uniform sampler2D diffuseTex;
uniform vec3 lightPos;
uniform mat4 modelMatrix;
uniform mat4 lightVP;
uniform float farPlane;
out vec4 worldPos; //for FS

void main()
{
    worldPos = modelMatrix * vec4(position, 1.0);
    gl_Position = lightVP * worldPos;
}