#version 430

in layout(location = 0) vec3 position;

uniform mat4 camProjMatrix;
uniform mat4 camViewMatrix;
uniform mat4 modelMatrix;
out vec4 pos;
//out vec2 texcoords;

void main()
{
	gl_Position = camProjMatrix * camViewMatrix * vec4(position, 1.0);
	pos = gl_Position;
	
	//texcoords = uv;
}