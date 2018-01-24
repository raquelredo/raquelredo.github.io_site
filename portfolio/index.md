---
layout: portfolio
title: "Portfolio"
permalink: /portfolio/
author_profile: true
comments: false
---
These are some of the projects that I have been coding on my own or while doing some DS Courses. You can access each individual coding just by clicking any of them.


<div class="grid__wrapper">
  {% for post in site.portfolio %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
