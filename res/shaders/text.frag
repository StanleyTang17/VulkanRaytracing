#version 450

layout(location = 0) in vec2 fragTexCoords;
layout(location = 1) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(binding = 3) uniform sampler2D fontTexture;

void main() {
	outColor = vec4(fragColor, texture(fontTexture, fragTexCoords).r);
	//outColor = vec4(1.0, 0.0, 0.0, 1.0);
}