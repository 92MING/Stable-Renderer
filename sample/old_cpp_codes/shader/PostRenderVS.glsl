#version 430

//data of quad
in layout(location=0) vec3 position;
in layout(location=1) vec2 uv;

uniform sampler2D screenColorTexture;
uniform sampler2D depthStencilTexture;

out vec4 color;

void main(){
    color = texture(screenColorTexture, uv);
    gl_Position = vec4(position.x, position.y, 0.0, 1.0); 
 }