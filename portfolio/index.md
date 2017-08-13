---
layout: page
title: Portfolio
excerpt: "Projects done on Data Science"
search_omit: true
---
The following projects have been done on my own or while doing some DS Courses. You can access each individual coding just by clicking any of tem.

<div class="grid__wrapper">
  {% for post in site.categories.portfolio %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
