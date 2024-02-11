import '../src/'
import React from 'react';
import Main from './components/main';

const App = () => (
  <div style={{ fontSize: '50px', textAlign: 'center' }}>
    <img src={require('./logo.png')} />

    <Main />
  </div>
);

export default App;