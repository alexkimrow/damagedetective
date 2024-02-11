import React, { useState } from 'react';
import axios from 'axios';


function ImageUpload() {
    const [file, setFile] = useState(null);
    const [image, setImage] = useState(null);
    const [responseText, setResponseText] = useState('');
    const handleClearResponse = () => {
        setResponseText('');
        setFile(null);
        setImage(null);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', file);

        if (file != null) {
            setImage(URL.createObjectURL(file))
        }

        try {
            const response = await axios.post('http://localhost:5000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            console.log(response.data);
            setResponseText(response.data);

        } catch (error) {
            console.error('Error uploading image:', error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <img src={image} />
            </div>
            <div style={{ fontSize: '20px', textAlign: 'center' }}>
                <input type="file" onChange={(event) => setFile(event.target.files[0])} />
                <button type="submit">Upload</button>
                <button onClick={handleClearResponse}>Clear</button>
                {responseText && <p>{responseText}</p>}
            </div>
        </form>
    );
}

export default ImageUpload;