// src/components/ChatBox.jsx
import React, { useEffect, useRef } from "react";
import Message from "./Message";

const ChatBox = ({ messages }) => {
    const chatRef = useRef(null);

    // Tự cuộn xuống khi có tin nhắn mới
    useEffect(() => {
        if (chatRef.current) {
            chatRef.current.scrollTop = chatRef.current.scrollHeight;
        }
    }, [messages]);

    return (
        <div ref={chatRef} style={{
            border: "1px solid #ccc",
            borderRadius: 15,
            padding: 15,
            height: 480,
            overflowY: "auto",
            backgroundColor: "#f0f8ff",
            display: "flex",
            flexDirection: "column",
            gap: 8
        }}>
            {messages.map((msg, idx) => <Message key={idx} sender={msg.sender} text={msg.text} />)}
        </div>
    );
};

export default ChatBox;
