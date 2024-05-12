// TODO: light system is not yet finished.

#version 430 core

in layout (location = 0) vec3 position;
uniform mat4 modelMatrix;
uniform mat4 lightSpaceMatrix;

void main()
{
    gl_Position = lightSpaceMatrix  * modelMatrix * vec4(position, 1.0);
}
