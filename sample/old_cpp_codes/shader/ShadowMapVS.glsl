#version 430
//vertex shader of Directional Light & Spot light

in layout (location = 0) vec3 position;

uniform mat4 lightVP;
uniform mat4 modelMatrix;

void main()
{
    gl_Position = lightVP * modelMatrix * vec4(position, 1.0);
}
