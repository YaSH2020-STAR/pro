import React, { useEffect, useRef } from 'react'
import { useState,Suspense } from 'react'
import sakura from "../assets/sakura.mp3";
import { Canvas } from '@react-three/fiber'
import Loader from '../components/Loader'
import { Bird, Island, Plane, Sky } from "../models"
import HomeInfo from '../components/HomeInfo';
import { soundoff, soundon } from '../assets/icons';
const Home = () => {
  const audioRef = useRef(new Audio(sakura));
  audioRef.current.volume = 0.4;
  audioRef.current.loop = true;

  const [currentStage, setCurrentStage] = useState(1);
  const [isRotating, setIsRotating] = useState(false);
  const [isPlayingMusic, setIsPlayingMusic] = useState(false);

  useEffect(() => {
    if (isPlayingMusic) {
      audioRef.current.play();
    }

    return () => {
      audioRef.current.pause();
    };
  }, [isPlayingMusic]);
 
  const adjustIslandForScreenSize = () => {

    let screenScale= null; 
    let screenPosition = [0, -18.5, -125];
    let rotation = [0.1, 4.7, 0];

    if(window.innerWidth <768) {
      screenScale = [0.9, 0.9, 0.9];
      
    }else {
      screenScale = [1, 1, 1];
      
    }
    return[screenScale, screenPosition, rotation];
  };

  const adjustPlaneForScreenSize = () => {
    let screenScale, screenPosition;


    if (window.innerWidth < 768) {
      screenScale = [1.5, 1.5, 1.5];
      screenPosition = [0, -1.5, 0]

    } else {
      screenScale = [3, 3, 3];
      screenPosition = [0, -4, -4]
    }
    return [screenScale, screenPosition];
  }

    const [islandScale, islandPosition, islandRotation] = adjustIslandForScreenSize();
    const [planeScale, planePosition] = adjustPlaneForScreenSize();

    return (
      <section className="w-full h-screen relative">
      <div className="absolute top-28 left-0 right-0 z-10 flex items-center justify-center">
        {currentStage && <HomeInfo currentStage={currentStage} />}
      </div> 

      
        <Canvas
          className={`w-full h-screen bg-transparent ${isRotating ?
            'cursor-grabbing' : 'cursor-grab'}`}
          camera={{ near: 0.1, far: 1000 }}
        >
          <Suspense fallback={<Loader />}>
            <directionalLight position={[1, 1, 1]} intensity={2} />
            <ambientLight intensity={0.7} />
            <spotLight />
            <hemisphereLight skyColor="#b1e1ff" groundColor="#000000" intensity={1} />

            <Bird />
            <Sky  isRotating={isRotating} />
            <Island
             isRotating={isRotating}
             setIsRotating={setIsRotating}
             setCurrentStage={setCurrentStage}
             position={islandPosition}
             rotation={[0.1, 4.7077, 0]}
             scale={islandScale}
            />
            <Plane
              isRotating={isRotating}
              planeScale={planeScale} 
              planePosition={planePosition}
              rotation={[0, 20, 0]}
            />
 
          </Suspense>

        </Canvas>
        <div className='absolute bottom-2 left-2'>
        <img
          src={!isPlayingMusic ? soundoff : soundon}
          alt='jukebox'
          onClick={() => setIsPlayingMusic(!isPlayingMusic)}
          className='w-10 h-10 cursor-pointer object-contain'
        />
      </div>
      </section>
    );
  };

export default Home;
