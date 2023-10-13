#version 430

in layout(location=0) vec3 pos;
in layout(location=1) vec2 uv;

out vec3 worldPos;
out vec3 worldNormal;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;
//out float depth;
uniform sampler2D waterDepthMap;

void main()
{
	worldPos = (modelMatrix * vec4(pos, 1.0)).xyz ;
	//depth = texture2D(waterDepthMap, uv).r;
	worldNormal = normalize((transpose(inverse(modelMatrix)) * vec4(vec3(0.0,1.0,0.0), 1.0)).xyz);
	gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(pos, 1.0);
}

