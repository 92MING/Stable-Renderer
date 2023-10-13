#version 430
//vertex shader for shadow cube map (point light)

in layout (location = 0) vec3 position;
in layout (location =2) vec2 uv;

uniform mat4 modelMatrix;
uniform mat4 lightSpaceMatrix;

out vec2 texCoords;
out vec4 fragPos;

void main()
{
    texCoords = uv;
    fragPos = modelMatrix * vec4(position, 1.0);
    gl_Position = lightSpaceMatrix * fragPos ;
}