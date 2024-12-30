import{ konverge, tech} from "../assets/images";
import {
    car,
    contact,
    css,
    git,
    github,
    linkedin,
    nodejs,
    ezee,
    cust,
    MachineLearning,
    NumPy,
    opencv,
    Python,
    react,
    tailwindcss,
    threads,
    typescript,
    mongodb,
    nextjs,
    sentiment,
    auto
} from "../assets/icons";

export const skills = [

    {
        imageUrl: Python,
        name: "Python",
        type: "Frontend",
    },

    
    {
        imageUrl: NumPy,
        name: "NumPy",
        type: "Frontend",
    },
   
    {
        imageUrl: opencv,
        name: "OprnCV",
        type: ".",
    },
    {
        imageUrl: github,
        name: "GitHub",
        type: "Version Control",
    },

  
    {
        imageUrl: mongodb,
        name: "MongoDB",
        type: "Database",
    },
    {
        imageUrl: nextjs,
        name: "Next.js",
        type: "Frontend",
    },
    {
        imageUrl: nodejs,
        name: "Node.js",
        type: "Backend",
    },
    {
        imageUrl: react,
        name: "React",
        type: "Frontend",
    },
    
    {
        imageUrl: tailwindcss,
        name: "Tailwind CSS",
        type: "Frontend",
    },
    
];

export const experiences = [
    {
        title: "Artificial Intelligence Intern",
        company_name: "Technobase IT Solutions",
        icon: tech,
        iconBg: "#accbe1",
        date: "Aug 2022 - Nov 2022",
        points: [
            "Devised an intelligent decision-making model within chatbot system, assessing over 200 process workflows, reducing response times, and restructuring operational efficiency in handling client queries.",
        ],
    },
    {
        title: "Data Science Intern",
        company_name: "Konverge.AI",
        icon: konverge,
        iconBg: "#fbc3bc",
        date: "Aug 2022 - Jul 2023",
        points: [
            "Designed and implemented a Python framework to analyze 10,000+ consumer profiles, boosting predictive capabilities and optimizing marketing strategies in wine sector.",
        ],
    },
    
];

export const socialLinks = [
    {
        name: 'Contact - Yash',
        iconUrl: contact,
        link: '/contact',
    },
    {
        name: 'GitHub - Yash',
        iconUrl: github,
        link: 'https://github.com/YourGitHubUsername',
    },
    {
        name: 'LinkedIn - Yash',
        iconUrl: linkedin,
        link: 'https://www.linkedin.com/in/yash-dorshetwar-55a983191/',
    }
    
];

export const projects = [
    {
        iconUrl: sentiment,
        theme: 'btn-back-red',
        name: 'Mental Health Monitoring Systems',
        description: 'Mental Health Monitoring System demonstrates the potential of combining advanced machine learning and NLP techniques to identify emotional states and provide meaningful interventions.',
        link: 'https://github.com/adrianhajdin/pricewise',
    },
    {
        iconUrl: cust,
        theme: 'btn-back-green',
        name: 'Customer Segmentation for Wine Industry',
        description: 'Executed LDA and clustering techniques to drive a 20% increase in targeted sales campaigns, using Python and machine learning algorithms enabling targeted marketing strategies.',
        link: 'https://github.com/adrianhajdin/threads',
    },
    {
        iconUrl: ezee,
        theme: 'btn-back-blue',
        name: 'Ezee : A Voice based chatbot for Automation',
        description: '•	Proposed Project is a performance-wise efficient virtual assistant for simple daily tasks in human life, based on concepts of internet of things, speech recognition, natural language processing, and artificial intelligence.',
        link: 'https://github.com/adrianhajdin/project_next13_car_showcase',
    },
    {
        iconUrl: auto,
        theme: 'btn-back-pink',
        name: 'Auto Attendance and Subject Recommendation System',
        description: 'Created a virtual assistant improving task completion efficiency by 30% through IoT and AI integration, streamlining attendance recording and saving educators over 15 hours per week.',
        link: 'https://github.com/adrianhajdin/social_media_app',
    },
    {
        iconUrl: ezee,
        theme: 'btn-back-black',
        name: 'Hand-Written Digit Recognition',
        description: 'Deployed computer modeling and image processing algorithms, building neural networks with high accuracy, applied image processing to commercial software, and customized features using NLP and Computer vision.',
        link: 'https://github.com/adrianhajdin/projects_realestate',
    }
];