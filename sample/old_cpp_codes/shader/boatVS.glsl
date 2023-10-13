#version 430

in layout(location=0) vec3 pos;
in layout(location=1) vec2 texcoord;

out vec3 position;
out vec2 uv;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

void main()
{
	uv = texcoord;
	position = pos;
	gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(pos, 1.0);
}

