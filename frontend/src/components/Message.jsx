// src/components/Message.jsx
import React from "react";

const Message = ({ sender, text }) => {
    const isBot = sender === "bot";
    return (
        <div style={{
            display: "flex",
            justifyContent: isBot ? "flex-start" : "flex-end",
            margin: "5px 0"
        }}>
            <div style={{
                maxWidth: "75%",
                padding: "12px 18px",
                borderRadius: 20,
                backgroundColor: isBot ? "#caf0f8" : "#90e0ef",
                color: "#000",
                fontSize: 14,
                lineHeight: 1.4,
                boxShadow: "0 2px 5px rgba(0,0,0,0.1)"
            }}>
                {text}
            </div>
        </div>
    );
};

export default Message;
