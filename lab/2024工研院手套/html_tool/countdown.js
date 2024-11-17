function initCountdown(targetDateStr) {
    const targetDate = new Date(targetDateStr);

    function updateCountdown() {
        const now = new Date();
        const timeDiff = targetDate - now;

        const countdownElement = document.getElementById("countdown");
        if (!countdownElement) {
            console.error("Countdown element not found!");
            return;
        }

        if (timeDiff > 0) {
            const days = Math.floor(timeDiff / (1000 * 60 * 60 * 24));
            const hours = Math.floor((timeDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
            const minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((timeDiff % (1000 * 60)) / 1000);

            countdownElement.textContent = 
                `剩餘: ${days}天 ${hours}小時 ${minutes}分 ${seconds}秒`;
        } else {
            countdownElement.textContent = "剩餘: 0天 0小時 0分 0秒";
        }
    }

    // 每秒更新一次倒數
    setInterval(updateCountdown, 1000);

    // 初始化顯示
    updateCountdown();
}
