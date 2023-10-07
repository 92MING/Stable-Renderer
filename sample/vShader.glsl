#version 430
layout (location=0) in vec3 position;
layout (location=1) in vec4 color;
out vec4 vertexColor;
uniform mat4 MVP;

void main()
{
    vec4 pos = vec4(position, 1.0);
    gl_Position = MVP * pos;
    vertexColor = color;
}