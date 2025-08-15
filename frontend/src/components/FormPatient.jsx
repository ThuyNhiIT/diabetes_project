import React, { useState } from "react";
import { api } from "../api/api";
import ChatBox from "./ChatBox";

import {
    Box,
    Typography,
    TextField,
    MenuItem,
    Button,
    Paper,
    Grid,
    CircularProgress,
    FormControl,
    InputLabel,
    Select,
    FormControlLabel,
    Checkbox
} from '@mui/material';
import MedicalInformationIcon from '@mui/icons-material/MedicalInformation';
import SendIcon from '@mui/icons-material/Send';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';

const FormPatient = () => {
    const [formData, setFormData] = useState({
        year: 2025,
        gender: "female",
        age: 45,
        location: "California",
        race_AfricanAmerican: 0,
        race_Asian: 0,
        race_Caucasian: 1,
        race_Hispanic: 0,
        race_Other: 0,
        hypertension: 0,
        heart_disease: 0,
        smoking_history: "current",
        bmi: 28.4,
        hbA1c_level: 6.2,
        blood_glucose_level: 125,
    });

    const [loading, setLoading] = useState(false);
    const [loadingAsk, setLoadingAsk] = useState(false);
    const [question, setQuestion] = useState("");
    const [messages, setMessages] = useState([
        { sender: "bot", text: "💉 Xin chào! Vui lòng nhập thông tin bệnh nhân để dự đoán khả năng mắc tiểu đường hoặc đặt câu hỏi về bệnh." }
    ]);

    const handleChange = (e) => {
        const { name, value, checked, type } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? (checked ? 1 : 0) : (isNaN(value) ? value : Number(value))
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setMessages(prev => [...prev, { sender: "user", text: "Đã gửi thông tin bệnh nhân" }]);

        try {
            const res = await api.post("/predict", formData);
            const botMsg = `Dự đoán: ${res.data.prediction === 1 ? "Có nguy cơ tiểu đường" : "Không nguy cơ tiểu đường"}\nXác suất: ${(res.data.probability * 100).toFixed(2)}%`;
            setMessages(prev => [...prev, { sender: "bot", text: botMsg }]);
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { sender: "bot", text: "Có lỗi xảy ra. Vui lòng thử lại!" }]);
        } finally {
            setLoading(false);
        }
    };

    const handleAsk = async () => {
        if (!question.trim()) return;
        setLoadingAsk(true);
        setMessages(prev => [...prev, { sender: "user", text: question }]);
        setQuestion("");

        try {
            const res = await api.post("/ask", { query: question });
            setMessages(prev => [...prev, { sender: "bot", text: res.data.answer }]);
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { sender: "bot", text: "Không thể trả lời câu hỏi." }]);
        } finally {
            setLoadingAsk(false);
        }
    };

    return (
        <Box sx={{ p: 4, maxWidth: '1200px', mx: 'auto' }}>
            <Grid container spacing={4}>
                {/* Form Section */}
                <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                            <MedicalInformationIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
                            <Typography variant="h4" component="h2" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                                Thông tin bệnh nhân
                            </Typography>
                        </Box>
                        <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 2 }}>
                            <Grid container spacing={3}>
                                {/* Personal Info */}
                                <Grid item xs={12}>
                                    <Typography variant="h6" sx={{ mb: 1, color: 'text.secondary' }}>Thông tin cá nhân</Typography>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} sm={6}>
                                            <TextField fullWidth label="Năm" type="number" name="year" value={formData.year} onChange={handleChange} required />
                                        </Grid>
                                        <Grid item xs={12} sm={6}>
                                            <TextField fullWidth label="Tuổi" type="number" name="age" value={formData.age} onChange={handleChange} required />
                                        </Grid>
                                        <Grid item xs={12} sm={6}>
                                            <FormControl fullWidth>
                                                <InputLabel>Giới tính</InputLabel>
                                                <Select name="gender" value={formData.gender} onChange={handleChange}>
                                                    <MenuItem value="female">Nữ</MenuItem>
                                                    <MenuItem value="male">Nam</MenuItem>
                                                </Select>
                                            </FormControl>
                                        </Grid>
                                        <Grid item xs={12} sm={6}>
                                            <TextField fullWidth label="Vị trí" name="location" value={formData.location} onChange={handleChange} />
                                        </Grid>
                                    </Grid>
                                </Grid>

                                {/* Race */}
                                <Grid item xs={12}>
                                    <Typography variant="h6" sx={{ mb: 1, mt: 2, color: 'text.secondary' }}>Chủng tộc</Typography>
                                    <Grid container spacing={1}>
                                        {[
                                            { label: "African American", name: "race_AfricanAmerican" },
                                            { label: "Asian", name: "race_Asian" },
                                            { label: "Caucasian", name: "race_Caucasian" },
                                            { label: "Hispanic", name: "race_Hispanic" },
                                            { label: "Other", name: "race_Other" },
                                        ].map((race) => (
                                            <Grid item xs={6} sm={4} key={race.name}>
                                                <FormControlLabel
                                                    control={<Checkbox checked={formData[race.name] === 1} onChange={handleChange} name={race.name} />}
                                                    label={race.label}
                                                />
                                            </Grid>
                                        ))}
                                    </Grid>
                                </Grid>

                                {/* Health Metrics */}
                                <Grid item xs={12}>
                                    <Typography variant="h6" sx={{ mb: 1, mt: 2, color: 'text.secondary' }}>Chỉ số sức khỏe</Typography>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} sm={6}>
                                            <TextField fullWidth label="BMI" type="number" step="0.1" name="bmi" value={formData.bmi} onChange={handleChange} />
                                        </Grid>
                                        <Grid item xs={12} sm={6}>
                                            <TextField fullWidth label="HbA1c (%)" type="number" step="0.1" name="hbA1c_level" value={formData.hbA1c_level} onChange={handleChange} />
                                        </Grid>
                                        <Grid item xs={12}>
                                            <TextField fullWidth label="Đường huyết (mg/dL)" type="number" name="blood_glucose_level" value={formData.blood_glucose_level} onChange={handleChange} />
                                        </Grid>
                                    </Grid>
                                </Grid>

                                {/* Medical History */}
                                <Grid item xs={12}>
                                    <Typography variant="h6" sx={{ mb: 1, mt: 2, color: 'text.secondary' }}>Tiền sử bệnh</Typography>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} sm={6}>
                                            <FormControlLabel control={<Checkbox checked={formData.hypertension === 1} onChange={handleChange} name="hypertension" />} label="Huyết áp cao" />
                                        </Grid>
                                        <Grid item xs={12} sm={6}>
                                            <FormControlLabel control={<Checkbox checked={formData.heart_disease === 1} onChange={handleChange} name="heart_disease" />} label="Bệnh tim" />
                                        </Grid>
                                        <Grid item xs={12}>
                                            <FormControl fullWidth>
                                                <InputLabel>Lịch sử hút thuốc</InputLabel>
                                                <Select name="smoking_history" value={formData.smoking_history} onChange={handleChange}>
                                                    <MenuItem value="never">Không bao giờ</MenuItem>
                                                    <MenuItem value="ever">Từng hút</MenuItem>
                                                    <MenuItem value="current">Hiện tại</MenuItem>
                                                </Select>
                                            </FormControl>
                                        </Grid>
                                    </Grid>
                                </Grid>
                            </Grid>
                            <Button type="submit" fullWidth variant="contained" endIcon={loading ? null : <SendIcon />} sx={{ mt: 4, py: 1.5 }} disabled={loading}>
                                {loading ? <CircularProgress size={24} color="inherit" /> : 'Dự đoán'}
                            </Button>
                        </Box>
                    </Paper>
                </Grid>

                {/* Chat Section */}
                <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 4, borderRadius: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                            <ChatBubbleOutlineIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
                            <Typography variant="h4" component="h2" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                                Kết quả dự đoán & Chat bot
                            </Typography>
                        </Box>
                        <Box sx={{ flexGrow: 1, minHeight: 400 }}>
                            <ChatBox messages={messages} />
                        </Box>
                        <Box sx={{ display: 'flex', mt: 2 }}>
                            <TextField
                                fullWidth
                                placeholder="Nhập câu hỏi về bệnh tiểu đường..."
                                value={question}
                                onChange={(e) => setQuestion(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                            />
                            <Button variant="contained" color="secondary" onClick={handleAsk} disabled={loadingAsk} sx={{ ml: 1 }}>
                                {loadingAsk ? <CircularProgress size={20} /> : "Gửi"}
                            </Button>
                        </Box>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default FormPatient;
