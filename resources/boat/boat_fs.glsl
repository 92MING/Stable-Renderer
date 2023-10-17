#version 330 core

in vec3 vertexPosition_modelspace;
in vec2 vertexUV;
in vec4 globalPos;

uniform sampler2D boatDiffuseTex;
// uniform sampler2D boatNormalTex;
// uniform sampler2D boatMetallicTex;
// uniform sampler2D boatRoughnessTex;
// uniform sampler2D boatAOTex;

out vec4 fragColor;

void main()
{
	// vec4 diffuseColor = texture(boatDiffuseTex, vertexUV);
	// fragColor = vec4(diffuseColor.rgb, 1.0);

	//float depth = globalPos.z + globalPos.w;
	//depth = 1 - (depth / (1.0 + depth));
	//float gamma = 2.2;
	//depth = pow(depth, 1.0 / gamma);

	float depth = (globalPos.z / globalPos.w) * 0.5 + 0.5;
	depth = (depth - 0.955) / (1.0 - 0.955);
	depth = 1 - depth;
	float gamma = 2.2;
	depth = pow(depth, 1.0 / gamma);
	fragColor = vec4(depth, depth, depth, 1.0);
}