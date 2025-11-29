// ---------------- PROFILE DROPDOWN -------------------
document.addEventListener("DOMContentLoaded", function() {
    console.log("MangosQueen Web App Loaded!");

    // Dropdown Animation
    const profileDropdown = document.getElementById("profileDropdown");
    const dropdownMenu = document.getElementById("dropdownMenu");

    profileDropdown.addEventListener("click", function(event) {
        event.preventDefault();
        dropdownMenu.classList.toggle("show");
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", function(event) {
        if (!profileDropdown.contains(event.target) && !dropdownMenu.contains(event.target)) {
            dropdownMenu.classList.remove("show");
        }
    });
});


// ---------------- LOGIN ALERT MESSAGES -------------------
document.addEventListener('DOMContentLoaded', function () {
    const alert = document.getElementById('dashboardFlash');
    if (alert) {
        alert.classList.add('show');
        setTimeout(() => {
            alert.classList.remove('show');
            alert.classList.add('fade');
        }, 1000); // Adjust time as needed
    }
});

// ---------------- LOGIN AND REGISTER ALERT MESSAGES -------------------
document.addEventListener('DOMContentLoaded', function () {
    const flashMessage = document.getElementById('flash-message');
    if (flashMessage) {
        flashMessage.classList.add('show'); // Trigger pop-in

        // Wait before fading out
        setTimeout(() => {
            flashMessage.classList.remove('show');
            flashMessage.classList.add('fade'); // Trigger pop-out

            // Fully remove after fade animation
            setTimeout(() => {
                flashMessage.remove();
            }, 200); // match fade duration in CSS
        }, 1500); // how long the alert stays visible before fading
    }
});

// ---------------- DASHBOARD DESIGN AND EFFECTS -------------------
window.addEventListener("DOMContentLoaded", () => {
    const title = document.querySelector(".dashboard-title");
    const text = title.textContent.trim();
    title.innerHTML = "";

    for (let i = 0; i < text.length; i++) {
        const span = document.createElement("span");
        span.textContent = text[i];
        title.appendChild(span);
    }
});








// ---------------------------- ABOUT PAGE ----------------------------
// JavaScript for MangoSqueen About Page

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Add fade-in animation to cards as they come into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    // Observe all cards
    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Make tab content height consistent
    function equalizeTabContentHeight() {
        if (window.innerWidth > 768) {
            const tabContents = document.querySelectorAll('.tab-pane');
            let maxHeight = 0;
            
            // Reset heights first
            tabContents.forEach(content => {
                content.style.minHeight = 'auto';
                const contentHeight = content.offsetHeight;
                if (contentHeight > maxHeight) {
                    maxHeight = contentHeight;
                }
            });

            // Set equal heights
            tabContents.forEach(content => {
                content.style.minHeight = maxHeight + 'px';
            });
        } else {
            // On mobile, reset heights
            document.querySelectorAll('.tab-pane').forEach(content => {
                content.style.minHeight = 'auto';
            });
        }
    }

    // Run on tab change and window resize
    document.querySelectorAll('a[data-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', equalizeTabContentHeight);
    });

    window.addEventListener('resize', equalizeTabContentHeight);
    
    // Run once on page load (after a short delay to ensure content is rendered)
    setTimeout(equalizeTabContentHeight, 500);
    
    // Search functionality for FAQs
    const faqSearch = document.getElementById('faqSearch');
    if (faqSearch) {
        faqSearch.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const faqItems = document.querySelectorAll('#faqAccordion .card');
            
            faqItems.forEach(item => {
                const question = item.querySelector('.btn-link').textContent.toLowerCase();
                const answer = item.querySelector('.card-body').textContent.toLowerCase();
                
                if (question.includes(searchTerm) || answer.includes(searchTerm)) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }

    // Counter animation for statistics
    function animateCounters() {
        document.querySelectorAll('.counter').forEach(counter => {
            const target = parseInt(counter.getAttribute('data-target'), 10);
            const duration = 2000; // 2 seconds
            const increment = target / (duration / 16);
            let current = 0;
            
            const updateCounter = () => {
                current += increment;
                if (current < target) {
                    counter.textContent = Math.floor(current).toLocaleString();
                    requestAnimationFrame(updateCounter);
                } else {
                    counter.textContent = target.toLocaleString();
                }
            };
            
            updateCounter();
        });
    }

    // Run counter animation when statistics section comes into view
    const statsSection = document.querySelector('.statistics-section');
    if (statsSection) {
        const statsObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                animateCounters();
                statsObserver.unobserve(statsSection);
            }
        }, { threshold: 0.5 });
        
        statsObserver.observe(statsSection);
    }
});
  