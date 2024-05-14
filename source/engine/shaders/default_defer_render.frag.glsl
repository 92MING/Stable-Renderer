#version 430 core
#define NON_AI_OBJ_MAP_INDEX 2048	// non-AI obj map index, since map size=k^2, k usually ~2-3, so 2048 is a safe number
#define BAKING_VISUAL_VAL 512  // just for visiualize the baking correspondmap, for better debug

uniform sampler2D currentColor;   // vec4, rgba
uniform usampler2D currentIDs;      // ivec4 (spriteID, material id, vertexID, 3D pixel index)
uniform sampler2D currentPos;     // vec3, world space position
uniform sampler2D currentNormalDepth; // vec4, (vec3(view space normal), depth)
uniform sampler2D currentNoises; // vec4, latent noise
uniform sampler2D currentCanny;

// flags
uniform int diffusion_disable;  // whether diffusion process is disabled
uniform int is_baking;        // whether we are baking   // 0: no, 1: yes

in vec2 uv;  // screen space UV

out vec4 FragColor;

void main() {
    
    FragColor = texture(currentColor, uv).rgba;

    if (is_baking == 1) {
        uvec4 current_id = texture(currentIDs, uv);
        if (current_id.x + current_id.y + current_id.z + current_id.w > 0) {    // object exists
            if (current_id.z != NON_AI_OBJ_MAP_INDEX) {    // AI object
                float ratio = 1.0 - clamp(current_id.w / float(BAKING_VISUAL_VAL * BAKING_VISUAL_VAL), 0.0, 1.0);
                vec3 col = vec3(1.0, 1.0, 1.0);
                if (ratio < 1.0 / 6.0) {
                    col.r = 1.0;
                    col.g = ratio * 6.0;
                    col.b = 0.0;
                } else if (ratio < 2.0 / 6.0) {
                    col.r = 1.0 - (ratio - 1.0 / 6.0) * 6.0;
                    col.g = 1.0;
                    col.b = 0.0;
                } else if (ratio < 3.0 / 6.0) {
                    col.r = 0.0;
                    col.g = 1.0;
                    col.b = (ratio - 2.0 / 6.0) * 6.0;
                } else if (ratio < 4.0 / 6.0) {
                    col.r = 0.0;
                    col.g = 1.0 - (ratio - 3.0 / 6.0) * 6.0;
                    col.b = 1.0;
                } else if (ratio < 5.0 / 6.0) {
                    col.r = (ratio - 4.0 / 6.0) * 6.0;
                    col.g = 0.0;
                    col.b = 1.0;
                } else {
                    col.r = 1.0;
                    col.g = 0.0;
                    col.b = 1.0 - (ratio - 5.0 / 6.0) * 6.0;
                }
                col = mix(FragColor.rgb, col, ratio) * ratio * 2.0;
                FragColor = vec4(col, 1.0);
            }
        }
    }
}
