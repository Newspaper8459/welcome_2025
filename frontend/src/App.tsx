import Header from './components/Header';
import Main from './components/Main';
import app  from './css/app.module.scss';


function App() {
  return (
    <>
      <Header />
      <div className={app.main}>
        <Main />
      </div>
    </>
  )
}

export default App
