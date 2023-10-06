#pragma once
#include "DataStructure.hpp"
#include "Engine.h"

// 繼承這個類，就可以監聽遊戲輸入
class InputHandler: public EngineEvents {
protected:
    InputHandler(){
        KeyCallback.AddListener(this,&InputHandler::OnKeyInput);
        MouseCallback.AddListener(this, &InputHandler::OnMouseInput);
        ScrollCallback.AddListener(this, &InputHandler::OnScrollInput);
        CursorPositionCallback.AddListener(this, &InputHandler::OnCursorPositionInput);
    }
    ~InputHandler() {
        KeyCallback.RemoveListener(this, &InputHandler::OnKeyInput);
        MouseCallback.RemoveListener(this, &InputHandler::OnMouseInput);
        ScrollCallback.RemoveListener(this, &InputHandler::OnScrollInput);
        CursorPositionCallback.RemoveListener(this, &InputHandler::OnCursorPositionInput);
    }
    
public:
    virtual void OnKeyInput(keyCallbackData data) {}
    virtual void OnMouseInput(mouseCallbackData data) {}
    virtual void OnScrollInput(scrollCallbackData data) {}
    virtual void OnCursorPositionInput(cursorPositionCallbackData data) {}
};