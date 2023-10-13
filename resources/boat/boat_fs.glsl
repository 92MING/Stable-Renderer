#version 330 core

in vec3 vertexPosition_modelspace;
in vec2 vertexUV;

uniform sampler2D boatDiffuseTex;
// uniform sampler2D boatNormalTex;
// uniform sampler2D boatMetallicTex;
// uniform sampler2D boatRoughnessTex;
// uniform sampler2D boatAOTex;

out vec4 fragColor;

void main()
{
	vec4 diffuseColor = texture(boatDiffuseTex, vertexUV);
	fragColor = vec4(diffuseColor.rgb, 1.0);
}